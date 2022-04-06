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

from parameters import get_parameters, setup_dom
from modules.density_extraction import density_extraction
from modules.NO2 import *


parser = argparse.ArgumentParser()
parser.add_argument("--do_ensemble", type=int, default=1, required=False)
parser.add_argument("--do_2dof", type=int, default=0, required=False)
parser.add_argument("--multiProc_ind", type=int, default=None, required=False)
args = parser.parse_args()



def main(data_parameters, return_extraction=False):

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



if __name__ == "__main__":

  data_parameters = get_parameters()

  q_max = [10]
  #q_max = [5, 7.5, 10, 12.5, 15, 17.5, 20]
  #sigmas = np.insert(1./(10**np.arange(11)), 0, 0.163)
  ston = [100]
  #ston = [25, 50, 100, 200, 400]
  lmk_arr = [[100, 100]]
  lmk_arr = [[25, 12.5], [25, 25], [25, 50], [25, 100], [12.5, 100], [50, 100]]
  options = []

  for bg in ston:
    for lg in lmk_arr:
      adm_params = copy(data_parameters["ADM_params"])
      adm_params["temperature"] = lg[0]
      adm_params["probe_FWHM"] = lg[1]
      for q in q_max:
        fit_range = [0.5, q]
        options.append({
          "fit_range"  : fit_range,
          "dom"        : np.linspace(0, q, int(500*(1+fit_range[0]/fit_range[1]))),
          "ADM_params" : copy(adm_params),
          "simulate_error" : ("StoN", (bg, [0.5,4]))})
          #"simulate_error" : ("constant_background", bg)})
          #"simulate_error" : ("constant_sigma", bg)})




  if args.multiProc_ind is not None:
    for k,v in options[args.multiProc_ind].items():
      data_parameters[k] = v
    data_parameters = setup_dom(data_parameters)
  #if not args.do_ensemble:
  #  data_parameters["molecule"] = data_parameters["molecule"] + "_single"
  #  #data_parameters["init_thetas_std_scale"] *= 1e-2

  main(data_parameters)
