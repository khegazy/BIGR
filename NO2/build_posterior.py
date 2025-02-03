import sys, os, glob, time
import argparse
import h5py
from copy import copy as copy
import numpy as np
import scipy as sp
import corner
import numpy.random as rnd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, Process, Manager
import matplotlib.pyplot as plt
from matplotlib import cm, lines

from parameters import get_parameters, setup_dom, setup_init_thetas
from modules.density_extraction import density_extraction
from modules.NO2 import *


parser = argparse.ArgumentParser()
parser.add_argument("--do_ensemble", type=int, default=1, required=False)
parser.add_argument("--do_2dof", type=int, default=0, required=False)
parser.add_argument("--multiProc_ind", type=int, default=None, required=False)
args = parser.parse_args()



def main(data_parameters, return_extraction=False):
  """
  This function initializes and runs the MCMC to retrieve the marginalized
  posterior after setting the input functions based upone the data_parameters.

      Parameters
      ----------
      data_parameters : dictionary
          The dictionary of runtime parameters used to define variables
          for both the density extraction and mode search

      Returns
      -------
  """

  #####  Setup ensemble/density generators and log prior  #####
  if "single" in data_parameters["molecule"]:
    input_ensemble_generator = single_molecule_generator
  else:
    input_ensemble_generator = molecule_ensemble_generator

  if "density_model" in data_parameters:
    if data_parameters["density_model"] == "PDF":
      input_density_generator = molecule_ensemble_generator
      if args.do_2dof or data_parameters["experiment"] == "2dof":
        input_log_prior = log_prior_2dof_gauss
      elif data_parameters["experiment"] == "3dof":
        input_log_prior = log_prior_3dof_gauss
      else:
        raise ValueError("Cannot handle experiment {}".format(
            data_parameters["experiment"]))
    
    elif data_parameters["density_model"] == "delta":
      input_density_generator = single_molecule_generator
      if args.do_2dof or data_parameters["experiment"] == "2dof":
        input_log_prior = log_prior_2dof_delta
      elif data_parameters["experiment"] == "3dof":
        input_log_prior = log_prior_3dof_delta
      else:
        raise ValueError("Cannot handle experiment {}".format(
            data_parameters["experiment"]))

    else:
      raise ValueError("Cannot handle density model {}".format(
          data_parameters["density_model"]))
  else:
    raise ValueError("Must provide default density generator")

      
  ####################################
  #####  Run Geometry Retrevial  #####
  ####################################

  extraction = density_extraction(data_parameters,
      get_molecule_init_geo,
      get_scattering_amplitudes,
      log_prior=input_log_prior,
      density_generator=input_density_generator,
      ensemble_generator=input_ensemble_generator,
      get_ADMs=get_ADMs)

  if return_extraction:
    return extraction

  #molecule = extraction.setup_calculations()

  """
  th = np.expand_dims(np.array([1.803-0.05, 1.118-0.05, 1.695-0.02]), 0)
  extraction.log_likelihood_density(th)
  sys.exit(0)
  """

  walkers_init = initialize_walkers(
      data_parameters, extraction.atom_positions)

  extraction.run_mcmc(walkers_init, data_parameters["run_limit"])


def update_parameters_cluster(data_parameters, index):

  base_q = 10.0
  q_max = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
  base_ston = 100
  ston = [12.5, 25, 50, 200, 400]
  base_lmk = [100, 100]
  lmk_arr = [[25, 12.5], [25, 25], [25, 50], [25, 100], [12.5, 100], [50, 100]]
  options = []

  for dist in ["PDF", "delta"]:
    for q in q_max:
      adm_params = copy(data_parameters["ADM_params"])
      adm_params["temperature"] = base_lmk[0]
      adm_params["probe_FWHM"] = base_lmk[1]
      options.append({
        "density_model" : dist,
        "fit_range"  : [0.5, q],
        "ADM_params" : copy(adm_params),
        "simulate_error" : ("StoN", (base_ston, [0.5,4]))})
      if q <= 5:
        print(len(options)-1)
        options[-1]["multiprocessing"] = 16
        options[-1]["min_acTime_steps"] = 2000
      if q <= 2.5:
        options[-1]["min_acTime_steps"] = 3000
    
    for q in q_max:
      if q == base_q:
        continue
      adm_params = copy(data_parameters["ADM_params"])
      adm_params["temperature"] = base_lmk[0]
      adm_params["probe_FWHM"] = base_lmk[1]
      options.append({
        "density_model" : dist,
        "fit_range"  : [0.5, q],
        "ADM_params" : copy(adm_params),
        "simulate_error" : ("StoN", (ston[-1], [0.5,4]))})
      if q <= 5:
        print(len(options)-1)
        options[-1]["multiprocessing"] = 16
        options[-1]["min_acTime_steps"] = 2000
      if q <= 2.5:
        options[-1]["min_acTime_steps"] = 3000

    for bg in ston:
      adm_params = copy(data_parameters["ADM_params"])
      adm_params["temperature"] = base_lmk[0]
      adm_params["probe_FWHM"] = base_lmk[1]
      options.append({
        "density_model" : dist,
        "fit_range"  : [0.5, base_q],
        "ADM_params" : copy(adm_params),
        "simulate_error" : ("StoN", (bg, [0.5,4]))})
      if bg <= 50:
        print(len(options)-1)
        options[-1]["multiprocessing"] = 16
        options[-1]["min_acTime_steps"] = 2000
      if bg <= 25:
        options[-1]["min_acTime_steps"] = 3000

    for lg in lmk_arr:
      adm_params = copy(data_parameters["ADM_params"])
      adm_params["temperature"] = lg[0]
      adm_params["probe_FWHM"] = lg[1]
      options.append({
        "density_model" : dist,
        "fit_range"  : [0.5, base_q],
        "ADM_params" : copy(adm_params),
        "simulate_error" : ("StoN", (base_ston, [0.5,4]))})

  if args.multiProc_ind > len(options) - 1:
    raise ValueError(
        "Index {} does not exist in options list that is {} long!".format(
            args.multiProc_ind, len(options)))
  for k,v in options[args.multiProc_ind].items():
    data_parameters[k] = v
  data_parameters = setup_dom(data_parameters)
  data_parameters = setup_init_thetas(data_parameters)

  # Do not use multiprocessing for delta distribution
  if data_parameters["density_model"] == "delta":
    if "multiprocessing" in data_parameters:
      del data_parameters["multiprocessing"]

  return data_parameters



if __name__ == "__main__":

  #####  Setup Method Parameters and Cluster Options  #####
  data_parameters = get_parameters()
  if args.multiProc_ind is not None:
    data_parameters = update_parameters_cluster(
        data_parameters, args.multiProc_ind)

  #####  Main  #####
  main(data_parameters)
