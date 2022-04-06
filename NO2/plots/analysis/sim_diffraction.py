import sys, os, glob, time
import h5py
import subprocess
from copy import copy as copy
from collections import defaultdict
import numpy as np
import scipy as sp

eval_times = np.linspace(37.742, 40.845, 8)

for tm in eval_times:
    fName = os.path.join("output",
        "NO2_symbreak_sim_diffraction-analytic_Qmax-20_time-{0:.6g}.h5".format(float(tm)))
    print("INFO: Looking for file " + fName)
    if not os.path.exists(fName):
        pp = subprocess.run("python ~/simulation/diffractionSimulation/diffraction.py --molecule NO2_symbreak --calculation_type analytic --xyz_file NO2_symbreak.xyz --basis_folder /cds/group/ued/scratch/N2O/axis_distributions/NO2/A/temp-100K --output_folder output --eval_time {}".format(tm),
                        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("\tSimulated")
        print(pp.stdout)
        print(pp.stderr)
        #p.wait()
        if not os.path.exists(fName):
            print("FAILED TO MAKE " + fName)
            sys.exit(0)
    else:
      print("\tFound")
