import os, sys
import numpy as np
from copy import copy

def get_parameters(run=None, molecule=None):

  data_parameters = {
      "molecule"           : "NO2_symbreak",
      "experiment"         : "3dof",
      "density_model"      : "PDF",
      "q_scale"            : 1.0,
      "wiener"             : False,
      "calc_type"          : 0,
      "multiprocessing"    : 10,
      "Nwalkers"           : 100,
      "run_limit"          : 100,
      "min_acTime_steps"   : 3000,
      "simulate_data"      : True,
      "simulate_error"     : ("StoN", (100, [0.5,4])),#("constant_sigma", 0.163)
      "N_mode_samples"     : 50,
      "mode_std_grid"      : np.array([-1, 0, 1]),
      "mode_tolerance"     : 0.0001,
      "plot_setup"         : True,
      "plot_progress"      : False,
      "dom"                : None,
      "fit_bases"          : None,
      "isMS"               : False,
      "fit_range"          : [0.5, 10],
      "elEnergy"           : 3.7e6,
      "sim_thetas"         : None,
      "init_thetas"        : None,
      "init_thetas_std_scale" : 0.002,
      "init_geo_xyz"       : None,
      "q_per_pix"          : 3.5/83,
      "scat_amps_dir"      : "/cds/home/k/khegazy/simulation/scatteringAmplitudes/3.7MeV/",
      "labels"             : None,#["d1", "d2", "angle"],#["d1", "d2", "angle"],
      "output_dir"         : "/cds/group/ued/scratch/khegazy/TeDDE/",
      "save_sim_data"      : "/cds/group/ued/scratch/khegazy/TeDDE/saved_simulations/",
      "ADM_params"         : { "folder"      : "/reg/data/ana15/ued/scratch/khegazy/TeDDE/axis_distributions",#"/cds/group/ued/scratch/N2O/axis_distributions",
                              "eval_times"    : np.linspace(37.5, 41.5, 100),
                              "probe_FWHM"    : 100,  # fs
                              "temperature"   : 100, # K
                              "intensity"     : 10,   # 10^12 W/cm^2
                              "normalize"     : False,
                              "subtract_mean" : False}
  }


  if molecule is not None:
    data_parameters["molecule"] = molecule

  # Setup XYZ file
  if data_parameters["molecule"] == "NO2":
      data_parameters["init_geo_xyz"] = "XYZ/NO2.xyz"
  elif data_parameters["molecule"] == "NO2_symbreak":
      data_parameters["init_geo_xyz"] = "XYZ/NO2_symbreak.xyz"

  # Setup multiproccessing and Nwalkers
  if "elta" in data_parameters["density_model"]:
    data_parameters["multiprocessing"] = 0 
    data_parameters["Nwalkers"] = 1000
    data_parameters["run_limit"] = 100
  else:
    data_parameters["multiprocessing"] = 10
    data_parameters["Nwalkers"] = 100
    data_parameters["run_limit"] = 100

  data_parameters = setup_dom(data_parameters)

  # Setup the LMK contributions used
  FB = []
  lmk = np.arange(4)*2
  for l in lmk:
    if l == 0 or l%2 != 0:
      continue
    for k in lmk:
      if l == lmk[-1] and k != 0:
        continue
      if k <= l:# and k > 0:
        FB.append(np.array([l,0,k]))
     #   FB.append(np.array([l,0,-1*k]))
     # elif k <= l:
     #   FB.append(np.array([l,0,k]))
  FB = np.array(FB).astype(int)
  data_parameters["fit_bases"] = FB

  if data_parameters["experiment"] == "3dof":
    data_parameters["sim_thetas"] = np.array(
        [1.35, 0.03, 1.05, 0.02, 2.34, 0.01])
    if data_parameters["density_model"] == "delta":
      data_parameters["init_thetas"] =\
          data_parameters["sim_thetas"][np.array([0,2,4], dtype=int)]
    elif data_parameters["density_model"] == "PDF":
      data_parameters["init_thetas"] = copy(data_parameters["sim_thetas"])
    if "single" in data_parameters["molecule"]:
      data_parameters["sim_thetas"] =\
          data_parameters["sim_thetas"][np.array([0,2,4], dtype=int)]
  elif data_parameters["experiment"] == "2dof":
    data_parameters["sim_thetas"] = np.array(
        [1.193, 0.02, 2.34, 0.01])
    if data_parameters["density_model"] == "delta":
      data_parameters["init_thetas"] =\
          data_parameters["sim_thetas"][np.array([0,2], dtype=int)]
    elif data_parameters["density_model"] == "PDF":
      data_parameters["init_thetas"] = copy(data_parameters["sim_thetas"])
    if "single" in data_parameters["molecule"]:
      data_parameters["sim_thetas"] =\
          data_parameters["sim_thetas"][np.array([0,2], dtype=int)]
  
  data_parameters["sim_thetas"] =\
      np.concatenate([data_parameters["sim_thetas"], [51]])

  # De Broglie wavelength angs
  C_AU = 1./0.0072973525664
  eV_to_au = 0.0367493
  angs_to_au = 1e-10/5.291772108e-11
  db_lambda = 2*np.pi*C_AU/\
      np.sqrt((data_parameters["elEnergy"]*eV_to_au + C_AU**2)**2\
      - (C_AU)**4) #au
  db_lambda /= angs_to_au  # angs
  k0 = 2*np.pi/db_lambda
  data_parameters["wavelength"] = db_lambda


  """
  detx, dety = np.meshgrid(
      np.linspace(-0.02, 0.02, 2*N-1), np.linspace(-0.02, 0.02, 2*N-1))
  det_dist = np.sqrt(4**2 + detx**2 + dety**2)
  """
  data_parameters["detector_dist"] = 1.#det_dist

  # Diffraction pattern simulation parameters
  data_parameters["beamline_length"] = 4
  
  return data_parameters

def setup_dom(data_parameters):
  # Setup dimension of measurement (dom)
  N = (data_parameters["fit_range"][1] - data_parameters["fit_range"][0])\
      /data_parameters["q_per_pix"]
  data_parameters["NradAzmBins"] = N
  data_parameters["dom"] = np.linspace(0, data_parameters["fit_range"][1],
      int(N*(1+data_parameters["fit_range"][0]/data_parameters["fit_range"][1])))

  return data_parameters


