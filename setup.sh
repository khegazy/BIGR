#!/usr/bin/env bash
#
# Set up a local BIGR working tree.
#
# The previous version of this script could not run: an unterminated quote on its last
# line meant bash could not parse it, it tested an undefined $FILE so two symlinks were
# silently skipped, `mkdir output` without -p failed on any re-run, and its two wget URLs
# pointed at a typo'd host (githubi.com) and at GitHub /blob/ pages that serve HTML rather
# than Python. See issues/012-setup-sh-broken.md.
#
# The downloads are gone: the two modules they fetched are now vendored in
# external_artifacts/ together with the electron scattering amplitudes, so nothing outside
# this repository has to exist.
#
# Usage:  bash setup.sh [experiment_dir]        (default: NO2)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="${1:-NO2}"
cd "$REPO"

if [ ! -d "$EXPERIMENT" ]; then
  echo "ERROR: experiment directory '$EXPERIMENT' not found in $REPO" >&2
  exit 1
fi

echo "INFO: Creating output directories"
mkdir -p plots XYZ output/logs output/saved_simulations
mkdir -p "$EXPERIMENT/plots"

echo "INFO: Creating symlinks for $EXPERIMENT"
# -n so that re-running does not nest a link inside an existing one
ln -sfn ../modules         "$EXPERIMENT/modules"
ln -sfn ../cpp_extensions  "$EXPERIMENT/cpp_extensions"

if [ -d "$EXPERIMENT/plots/analysis" ]; then
  echo "INFO: Creating symlinks for plots/analysis"
  ln -sfn ../../parameters.py      "$EXPERIMENT/plots/analysis/parameters.py"
  ln -sfn ../../XYZ                "$EXPERIMENT/plots/analysis/XYZ"
  ln -sfn ../../../modules         "$EXPERIMENT/plots/analysis/modules"
  ln -sfn ../../../cpp_extensions  "$EXPERIMENT/plots/analysis/cpp_extensions"
fi

echo "INFO: Checking vendored run-time dependencies"
for f in external_artifacts/modules/fitting.py \
         external_artifacts/modules/diffraction_simulation.py \
         external_artifacts/scattering_amplitudes/3.7MeV/nitrogen_dcs.dat \
         external_artifacts/scattering_amplitudes/3.7MeV/oxygen_dcs.dat; do
  if [ -f "$f" ]; then
    echo "        found $f"
  else
    echo "        WARNING: missing $f -- see external_artifacts/README.md" >&2
  fi
done

echo "INFO: Building the C++ extension"
# The committed .so was a Linux x86-64 binary and is no longer tracked, so this must be
# built for the host platform. Report the outcome honestly -- the old script printed
# "compiled correctly" unconditionally, even on failure.
if (cd cpp_extensions/lib && make clean >/dev/null 2>&1; make); then
  echo "INFO: C++ extension built. Use calc_type = 0 (recommended)."
  file cpp_extensions/lib/c_calc_extensions.so 2>/dev/null || true
else
  echo "WARNING: the C++ build failed." >&2
  echo "         calc_type = 1 (scipy) is a working fallback; calc_type = 2 is broken." >&2
  echo "         See issues/004-calc-type-1-and-2-broken.md" >&2
fi

echo
echo "INFO: Setup complete. Remaining steps:"
echo "  1. Install dependencies, e.g."
echo "       uv pip install --python .venv/bin/python \\"
echo "           numpy scipy matplotlib h5py emcee corner tqdm ipykernel nbconvert"
echo "  2. Stage the ADMs into the layout get_ADMs reads:"
echo "       .venv/bin/python scripts/stage_adms.py"
echo "  3. Check the forward model:"
echo "       MPLBACKEND=Agg .venv/bin/python scripts/test_physics.py"
echo "  4. Run the analysis (from inside $EXPERIMENT/ -- the .so path is cwd-relative):"
echo "       cd $EXPERIMENT && MPLBACKEND=Agg ../.venv/bin/python build_posterior.py"
echo
echo "  See how_to_run.md for the full procedure and the HDF5 formats."
