import os, sys
import numpy as np

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



N = 500 
rng = [0.01,10]
data_parameters = {
    "molecule"           : "NO2",#"NO2_symbreak",
    "experiment"         : "3dof",
    "q_scale"            : 1.0,
    "multiprocessing"    : True,
    "Nwalkers"           : 1000,
    "run_limit"          : 500,
    #"min_acTime_steps"  : 500,
    "simulate_data"      : True,
    "simulate_error"     : ("constant_sigma", 0.1),#("StoN", (350., [0.5,4])),#("constant_sigma", 0.1),# ("constant_background", 10),
    #"data_fileName"      : os.path.join(
    #      "../physics_time_basis/output/UED/NO2/",
    #      "UED_fit_results_Temp-100.0-100.0-1_Ints-10.0-10.0-1.h5"),
    "dom"               : np.linspace(0,rng[1],int(N*(1+rng[0]/rng[1]))),
    "fit_bases"         : FB,
    "isMS"              : False,
    "fit_range"         : [rng[0], rng[1]],
    "elEnergy"          : 3.7e6,
    "init_geo_xyz"      : "XYZ/NO2.xyz",#"XYZ/NO2_symbreak.xyz", 
    "scat_amps_path"    : "./scatteringAmplitudes/3.7MeV/",
    "labels"            : ["d1", "d2", "angle"],#["d1", "d2", "angle"],
    "output_dir"        : "/cds/group/ued/scratch/khegazy/TeDDE/",
    "ADM_kwargs"        : { "folder"      : "/cds/group/ued/scratch/N2O/axis_distributions",
                            "eval_times"  : np.linspace(37.5, 41.5, 100),
                            "probe_FWHM"  : 100,  # fs
                            "temperature" : 100,  # K
                            "intensity"   : 10}   # 10^12 W/cm^2
}


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


