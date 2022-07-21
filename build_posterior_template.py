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
parser.add_argument("--multiProc_ind", type=int, default=None, required=False)
args = parser.parse_args()



##################
#####  Main  #####
##################

def main(data_parameters, return_extraction=False):
  """
  This function initializes the density_extraction class based in input 
  runtime arguments and subsequently runs the MCMC. The MCMC builds the 
  posterior distribution P(Theta|C) based on the provided data/simulation, 
  or based on the simulations done in the density_extraction class. The 
  results are periodically saved, allowing one to kill and restart this 
  process without losing progress in the event of a job being killed.

      Parameters
      ----------
      data_params : dictionary
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
      input_log_prior = log_prior_gauss
    
    elif data_parameters["density_model"] == "delta":
      input_density_generator = single_molecule_generator
      input_log_prior = log_prior_delta

    else:
      raise ValueError("Cannot handle density model {}".format(
          data_parameters["density_model"]))
  else:
    raise ValueError("Must provide default density generator")



  #####  Run Geometry Retrevial  #####

  extraction = density_extraction(data_parameters,
      get_molecule_init_geo,
      get_scattering_amplitudes,
      log_prior=input_log_prior,
      density_generator=input_density_generator,
      ensemble_generator=input_ensemble_generator,
      get_ADMs=get_ADMs)

  if return_extraction:
    return extraction


  walkers_init = initialize_walkers(
      data_parameters, extraction.atom_positions)

  extraction.run_mcmc(walkers_init, data_parameters["run_limit"])




if __name__ == "__main__":

  ###  Get model parameters  ###
  data_parameters = get_parameters()

  ###  Build dictionary of various parameter combinations to run  ###
  q_max = [10]
  ston  = [25, 50, 100, 200, 400]
  lmk_arr = [[100, 100]]
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

  # Select which parameters based on runtime argument
  if args.multiProc_ind is not None:
    for k,v in options[args.multiProc_ind].items():
      data_parameters[k] = v
    data_parameters = setup_dom(data_parameters)

  #####  Run Main  #####
  main(data_parameters)
