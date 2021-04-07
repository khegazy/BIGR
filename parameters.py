import os
import numpy as np

FB = []
lmk = np.arange(4)*2
for l in lmk:
  if l == 0:
    continue
  for k in lmk:
    if k > l or (l > lmk[-2] and k>0):
      continue
    FB.append(np.array([l,0,k]))
FB = np.array(FB).astype(int)

N = 500
rng = [0.01,20]
data_parameters = {
    "molecule"           : "validation_molecule",
    "experiment"         : "UED",
    "q_scale"            : 1.0,
    "multiprocessing"   : True,
    "Nwalkers"          : 1000,
    "run_limit"          : 500,
    #"min_acTime_steps"  : 500,
    "posterior"         : "optimal",
    "simulate_data"      : True,
    #"simulate_error"     : ("constant", 10),
    #"data_fileName"      : os.path.join(
    #      "../physics_time_basis/output/UED/NO2/",
    #      "UED_fit_results_Temp-100.0-100.0-1_Ints-10.0-10.0-1.h5"),
    "dom"               : np.linspace(rng[0],rng[1],N),
    "fit_bases"         : FB,
    "isMS"              : False,
    "fit_range"         : [rng[0], rng[1]],
    "elEnergy"          : 3.7e6,
    "init_geo_xyz"      : "XYZ/NO2.xyz", 
    "scat_amps_path"    : "./scatteringAmplitudes/3.7MeV/"
}