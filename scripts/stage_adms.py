"""Stage the NO2 axis distribution moments (ADMs) into the layout ``get_ADMs`` reads.

The ADMs shipped in ``NO2_properties_simulation/axis_dist/`` are already ``.npy``, but they
sit in a different directory layout than the one ``modules/NO2.py:get_ADMs`` expects:

    source                                          target (what get_ADMs globs)
    A/temp-{T}K/A-Mat_NO2{word}K I=..TW tau=..fs D{L}{K}.npy
                                             ->  <root>/NO2/ADMs/temp-{T}K/{I}TW_{FWHM}fs/<same name>
    times/temp_{T}K.npy                      ->  <root>/NO2/ADMs/temp-{T}K/{I}TW_{FWHM}fs/times.npy

No numerical conversion happens here -- this is purely a reshape of the directory tree, so
``get_ADMs`` itself is left untouched. The basis filenames are deliberately preserved because
``get_ADMs`` parses the angular momentum indices positionally from the last characters of the
path, ``L = int(fl[-6])`` and ``K = int(fl[-5])`` (modules/NO2.py:558-559): for
``... D62.npy`` that yields L=6, K=2. Per ``axis_dist/README.txt`` the ``D{K}{S}`` suffix is
the ADM index itself ("the D22 file is the ADM A^2_02"), matching the code's ``[L, 0, K]``
convention.

Usage
-----
    python scripts/stage_adms.py                 # stage every temperature found
    python scripts/stage_adms.py --temps 30      # stage only 30 K
"""

import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "NO2_properties_simulation", "axis_dist")
DEST_ROOT = os.path.join(SRC, "bigr_layout")

# Both are encoded in the source filenames ("I=10TW tau=100fs") rather than the directory
# tree, so they are read back out of the filenames and used to build the target directory.
FNAME_RE = re.compile(r"I=(?P<intensity>[\d.]+)TW tau=(?P<fwhm>[\d.]+)fs D(?P<L>\d)(?P<K>\d)\.npy$")


def format_temp(temperature: float) -> str:
    """Reproduce get_ADMs' temperature directory spelling (modules/NO2.py:531-534)."""
    if int(temperature * 10) % 10 == 0:
        return str(int(temperature))
    return "{0:1g}".format(temperature)


def format_fwhm(fwhm: float) -> str:
    """Reproduce get_ADMs' pulse-duration spelling (modules/NO2.py:527-530)."""
    if int(fwhm * 10) % 10 == 0:
        return str(int(fwhm))
    return "{0:1g}".format(fwhm)


def stage_temperature(temp_dir: str) -> tuple[str, int]:
    """Stage one ``A/temp-{T}K`` directory. Returns (target_dir, n_bases_copied)."""
    temp_label = os.path.basename(temp_dir)                  # e.g. "temp-30K"
    temperature = float(temp_label.replace("temp-", "").rstrip("K"))

    bases = sorted(glob.glob(os.path.join(temp_dir, "A*.npy")))
    if not bases:
        raise FileNotFoundError("no A*.npy bases in {}".format(temp_dir))

    match = FNAME_RE.search(os.path.basename(bases[0]))
    if match is None:
        raise ValueError("cannot parse intensity/FWHM from {}".format(bases[0]))
    intensity = float(match.group("intensity"))
    fwhm = float(match.group("fwhm"))

    # get_ADMs builds: <folder>/NO2/ADMs/temp-{T}K/{int(I)}TW_{FWHM}fs/  (modules/NO2.py:536-541)
    target = os.path.join(
        DEST_ROOT, "NO2", "ADMs",
        "temp-{}K".format(format_temp(temperature)),
        "{}TW_{}fs".format(int(intensity), format_fwhm(fwhm)),
    )
    os.makedirs(target, exist_ok=True)

    for basis in bases:
        shutil.copy2(basis, os.path.join(target, os.path.basename(basis)))

    # get_ADMs opens a literal "times.npy" inside the same directory (modules/NO2.py:545).
    times_src = os.path.join(SRC, "times", "temp_{}K.npy".format(format_temp(temperature)))
    if not os.path.exists(times_src):
        raise FileNotFoundError("missing time axis {}".format(times_src))
    shutil.copy2(times_src, os.path.join(target, "times.npy"))

    times = np.load(os.path.join(target, "times.npy"))
    print("  {}  <- {} bases, times {} spanning [{:.2f}, {:.2f}] ps".format(
        os.path.relpath(target, REPO), len(bases), times.shape, times.min(), times.max()))
    return target, len(bases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temps", nargs="*", default=None,
                        help="temperatures to stage (e.g. 30 20); default is all found")
    args = parser.parse_args()

    temp_dirs = sorted(glob.glob(os.path.join(SRC, "A", "temp-*K")))
    if args.temps:
        wanted = {"temp-{}K".format(t) for t in args.temps}
        temp_dirs = [d for d in temp_dirs if os.path.basename(d) in wanted]
    if not temp_dirs:
        print("ERROR: no matching temperature directories under {}".format(SRC), file=sys.stderr)
        return 1

    print("Staging ADMs into {}".format(os.path.relpath(DEST_ROOT, REPO)))
    for temp_dir in temp_dirs:
        stage_temperature(temp_dir)
    print("\nSet ADM_params['folder'] to: {}".format(os.path.relpath(DEST_ROOT, REPO)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
