import os, sys
import numpy as np
from copy import copy

# Repo-relative paths, derived from this file's location so they resolve no matter which
# directory the analysis is launched from (the scripts run from NO2/, notebooks may not).
_HERE = os.path.dirname(os.path.abspath(__file__))      # .../BIGR/NO2
_REPO = os.path.dirname(_HERE)                          # .../BIGR
_ART  = os.path.join(_REPO, "external_artifacts")


def get_parameters(run=None, molecule=None):

  data_parameters = {
      # NO2_symbreak = the symmetry-broken (asymmetric) geometry, N-O = 1.35 and 1.05 Ang,
      # so each bond carries its own degree of freedom -> experiment "3dof". The symmetric
      # geometry (XYZ/NO2.xyz, both bonds 1.1934 Ang) shares one bond length and belongs
      # with experiment "2dof". See how_to_run.md; the two must always be kept consistent.
      "molecule"           : "NO2_symbreak",
      "experiment"         : "3dof",
      "density_model"      : "PDF",
      "q_scale"            : 1.0,
      "wiener"             : False,
      "calc_type"          : 0,
      # NOTE: multiprocessing / Nwalkers / run_limit set here are OVERWRITTEN further down
      # by the density_model branch -- edit those lines, not these.
      "multiprocessing"    : 0,
      "Nwalkers"           : 16,
      "run_limit"          : 50,
      "min_acTime_steps"   : 5,
      "max_iterations"     : 1000,
      "simulate_data"      : True,
      # Error model. constant_sigma is the default here because the StoN (Poissonian)
      # model, although it runs end to end, cannot produce a usable signal-to-noise with
      # the ADMs available in this repo -- see how_to_run.md, "Why not StoN". sigma = 0.163
      # was already the alternative suggested in this file and is the right scale: it puts
      # the 1-sigma resolution on the N-O distances at a few mAngstrom.
      "simulate_error"     : ("constant_sigma", 0.163),#("StoN", (100, [0.5,4]))
      "N_mode_samples"     : 25,
      "mode_std_grid"      : np.array([-1, 0, 1]),
      "mode_tolerance"     : 0.01,
      "plot_setup"         : True,
      "plot_progress"      : False,
      "dom"                : None,
      "fit_bases"          : None,
      "isMS"               : False,
      "fit_range"          : [0.5, 5],
      "elEnergy"           : 3.7e6,
      "sim_thetas"         : None,
      "init_thetas"        : None,
      "init_thetas_std_scale" : 0.002,
      "init_geo_xyz"       : None,
      "q_per_pix"          : 2*3.5/83,
      "scat_amps_dir"      : os.path.join(_ART, "scattering_amplitudes", "3.7MeV"),
      "labels"             : None,#["d1", "d2", "angle"],#["d1", "d2", "angle"],
      "output_dir"         : os.path.join(_REPO, "output"),
      "save_sim_data"      : os.path.join(_REPO, "output", "saved_simulations"),
      # ADMs: generate this tree with `python scripts/stage_adms.py`. Available
      # temperatures are 1, 10, 20, 30 and 300 K, and the time axis spans only
      # -0.2 to 40.3 ps, so eval_times must stay inside that window.
      #
      # eval_times brackets the rotational revival near 18 ps, where the ensemble
      # anisotropy peaks. This matters a lot: simulate_error_StoN propagates the C
      # coefficient errors through inv(A^T W A) with the mean-subtracted ADMs as A, so a
      # window with little anisotropy gives a nearly singular A and enormous error bars.
      # Likewise temperature 1 K rather than 30 K -- the L=6 ADM is ~21x larger cold
      # (0.036 vs 0.0017), which is what makes the higher-order C_lmk measurable at all.
      # WARNING: eval_times and probe_FWHM are NOT encoded in get_fileName, so delete the
      # save_sim_data cache after changing them or stale coefficients are silently reused.
      "ADM_params"         : { "folder"      : os.path.join(
                                  _REPO, "NO2_properties_simulation",
                                  "axis_dist", "bigr_layout"),
                              "eval_times"    : np.linspace(16.0, 20.0, 25),
                              "probe_FWHM"    : 100,  # fs
                              "temperature"   : 1,   # K
                              "intensity"     : 10,   # 10^12 W/cm^2
                              "normalize"     : False,
                              "subtract_mean" : False}
  }


  if molecule is not None:
    data_parameters["molecule"] = molecule

  # Setup XYZ file
  if data_parameters["molecule"] == "NO2":
      data_parameters["init_geo_xyz"] = os.path.join(_HERE, "XYZ", "NO2.xyz")
  elif data_parameters["molecule"] == "NO2_symbreak":
      data_parameters["init_geo_xyz"] = os.path.join(_HERE, "XYZ", "NO2_symbreak.xyz")

  # Setup multiproccessing and Nwalkers.
  # multiprocessing MUST stay 0/1 on macOS: calculate_c_ensemble_multiProc pickles a bound
  # method whose self.spherical_j is a closure created in setup_calculations, which the
  # spawn start method cannot pickle. 0 keeps the fast in-process C++ path.
  if "elta" in data_parameters["density_model"]:
    data_parameters["multiprocessing"] = 0
    data_parameters["Nwalkers"] = 32
    data_parameters["run_limit"] = 50
  else:
    data_parameters["multiprocessing"] = 0
    data_parameters["Nwalkers"] = 32
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


