# 011 — `analyze_results.ipynb`: stale duplicate functions, undefined names, cluster paths

**Severity** P2 (a cell can produce a wrong figure rather than an error)
**Area** notebook
**Status** partly fixed — it now opens and the fast-run section works; the original sweep sections
still have these problems

Cell indices below are for the **current** notebook, which has 6 fast-run cells inserted after
cell 2 (so original cell *n* is now *n + 6*).

## What was already fixed

Recorded in `CHANGELOG.md`: the cell-0 `ImportError` (`calc_ensemble_dists`, `calc_dists` do not
exist), the removed `IPython.core.display` import, the optional `diffraction_simulation` import, and
the kernelspec that pinned a nonexistent `emcee_env` kernel.

## 11a. Two cells shadow `plot_functions` with stale signatures — **silent wrong-figure risk**

Cells **18** and **21** redefine functions already imported by `from modules.plot_functions import *`
in cell 0, with signatures that no longer match the call sites further down:

| Function | Notebook redefinition | `modules/plot_functions.py` | Call sites expect |
|---|---|---|---|
| `column_compare_dists` | cell 18: `(samples, ranges, labels, xticks=None, cn=cn_gauss)` | `:469` `(samples, centers, ranges, labels, xticks=None)` | the module version |
| `plot_trends_single` | cell 21: `(x, precision, modes, x_label, corrs=None, centers=cn_gauss, labels=…)` — **no `colors` kwarg** | `:364` `(…, colors=['b','r','k'])` | the module version (they pass `colors=`) |

`plot_trends_single` will at least raise on the unexpected `colors=`. `column_compare_dists` is the
dangerous one: the notebook version's second positional parameter is `ranges` while the module
version's is `centers`, so a call like `column_compare_dists([...], cn2, ranges2, labels2)` binds
`cn2` to `ranges` — **it will happily draw a figure with the axes ranges and the distribution
centres swapped.** No exception.

There is a third copy of `column_compare_dists` at cell **89**.

**Do not run cells 18 or 21.** A markdown cell now says so, but the real fix is to delete both
redefinitions and rely on the module.

## 11b. Undefined names

| Cell | Name | Note |
|---|---|---|
| 11 | `input_initialize_walkers` | never defined; the real name is `initialize_walkers` (`modules/NO2.py:225`). This cell duplicates `build_posterior.py` anyway — skip it |
| 65 | `exx.wiener` | `density_extraction` never sets a `wiener` attribute. Only `data_params["wiener"]` exists (`parameters.py:23`) → `AttributeError` |
| 19, 31, 93 | `ranges_theta` | `plot_2dproj_column` references `ranges_theta`, `cn` and `labels` as globals that are never assigned. Broken in **both** the notebook copy and `modules/plot_functions.py:643` |
| 23 | `get_CI` | function body is unfinished — starts with a bare `lsearch = 144747` and never returns |

## 11c. Hardcoded cluster paths and external scripts

| Cell | Depends on |
|---|---|
| 57 | `/cds/group/ued/scratch/khegazy/TeDDE/axis_distributions/NO2/ADMs/...` read directly as `.npy` |
| 62, 63 | shells out to `python ../diffraction.py --basis_folder /cds/group/ued/scratch/N2O/...`, then reads `output/NO2_sim_diffraction-analytic_Qmax-15_time-*.h5` |

`diffraction.py` is not in this repo. These cells produce the ADM and alignment figures and cannot
run without both the external script and those directories.

## 11d. The N₂O measured-data section needs a file that is not here

The section beginning around cell 28 does
`from parameters_N2O_data import get_parameters as get_parameters_data`.
`setup.sh:1,16` was supposed to symlink `parameters_N2O_data.py` from
`/cds/home/k/khegazy/analysis/2015/.../parameters.py`, but that symlink never worked
([012](012-setup-sh-broken.md)) and the measured N₂O dataset is not in the repo. Per the paper's
Data Availability statement the UED N₂O data is available from the authors on request, so this
section is expected to be unrunnable for outside users — worth stating in the notebook.

## 11e. The sweep sections each need their own completed MCMC run

The S/N, q-range and ADM-temperature scans read 20–28 HDF5 files each (7 configurations × 2 density
models, ×2 files). Every one requires a separate `build_posterior.py` + `mode_search.py` pair. At
the paper's settings that is hours to days per configuration.

The stored outputs in those cells are from the authors' original cluster runs. They were
**deliberately preserved**, and a markdown cell now states that they do not correspond to anything
in `output/` in this checkout.

## Suggested fix

1. Delete the three stale redefinitions (cells 18, 21, 89) — highest value, since 11a can silently
   mislead.
2. Delete or repair cells 11, 23, and the `exx.wiener` use in 65.
3. Fix `plot_2dproj_column` in `modules/plot_functions.py:643` to take `ranges`, `centers` and
   `labels` as arguments instead of globals, then drop the notebook copy.
4. Parameterise the `/cds/` paths in cells 57/62/63 the way `parameters.py` now does (repo-relative,
   derived from `__file__`).
5. Consider splitting the notebook: `analyze_results.ipynb` for the fast/current run, and
   `paper_figures.ipynb` for the sweeps. The current file is 10.5 MB, almost entirely stored output
   images, which makes it slow to open and awkward to diff.
