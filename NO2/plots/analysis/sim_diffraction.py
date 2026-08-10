"""Regenerate simulated diffraction images for the analysis-step figures.

    NOT FUNCTIONAL IN THIS REPOSITORY, AND NOT NEEDED TO RUN BIGR.

This script is not on the build_posterior.py -> mode_search.py path; nothing in the
retrieval imports it. It shells out (line ~30) to an external diffraction simulator at
`~/simulation/diffractionSimulation/diffraction.py` and to ADMs under `/cds/group/ued/...`
on the SLAC cluster. Neither path exists outside the original environment, so the
subprocess fails and the script exits.

To use it you would need to: install the `diffractionSimulation` package (see
NO2_properties_simulation/diffraction/README.md), replace the `~/simulation/...` path
below, and repoint --basis_folder at the in-repo ADMs, e.g.
    ../../../NO2_properties_simulation/axis_dist/A/temp-1K
Note the hardcoded temp-100K does not exist in this repository's ADM set -- only
1, 10, 20, 30 and 300 K -- so it cannot be a straight substitution.

See how_to_run.md for the analysis that does work.
"""

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
