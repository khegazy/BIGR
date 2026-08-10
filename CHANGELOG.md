# Changelog

All notable changes to BIGR are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

To run the code, read [`how_to_run.md`](how_to_run.md) — you do not need this file. Read this when
you want to know what changed, or why a line looks the way it does. Open defects are catalogued in
[`issues/`](issues/).

## [v0.0.2]

Revival release: as committed, `v0.0.1` could not run anywhere except the SLAC/LCLS cluster it was
written on. It now runs from a fresh clone on macOS/Linux, and the retrieval is verified against
known truth.

**The headline is a physics bug.** A centre-of-mass error scaled every molecule by 1/total_mass —
46× for NO₂ — corrupting every C coefficient in every calculation. It also explains a string of
symptoms previously recorded as separate defects. Any quantitative result produced with `v0.0.1`
should be recomputed.

### Fixed

- **Centre-of-mass bug in `rotate_to_principalI` / `rotate_to_principalI_ensemble`.** The code
  computed `r/M − Σmⱼrⱼ/M` instead of `r − Σmⱼrⱼ/M`, dividing every coordinate by the total mass, so
  all pairwise distances came out 46× too small for NO₂ (0.029 Å instead of 1.35 Å). This corrupted
  every C coefficient and pushed the C++ spherical-Bessel recursion into its numerically unstable
  regime. Fixing it resolved, in one change: the anomalously small L = 4 coefficients (now
  |C₄₀₀| = 2.95 > |C₆₀₀| = 1.94, as the physics requires), the apparent non-convergence of the
  ensemble quadrature, the unusably low `StoN` signal-to-noise, and every failed retrieval.
- **Measured-data import.** Called a non-existent `self.fig_I0`; behind that, `fit_I0` reshaped six
  values into a `(1,1)` array. Both fixed, and covered by a round-trip test.
- **`("data", …)` error model.** Had never run: an undefined name in the NaN-trimming loop, a float
  used as an array index, an off-by-one back-fill, and no termination guard on an all-NaN row.
- **Per-LMK diagnostic plots were all identical.** Both plotting loops built the filename from the
  loop variable but indexed the data at a hardcoded `[0]`, so all six files were byte-identical
  copies of the (2,0,0) curve. Diagnostic only — the noise added to the coefficients was always
  correct — but it made per-channel defects invisible.
- **`setup.sh` could not run at all**: an unterminated quote made it unparseable, two symlinks were
  guarded by an undefined variable, one download host was a typo, both download URLs served HTML
  rather than Python, and `mkdir` without `-p` failed on re-run. Rewritten and verified idempotent.
- **Simulated-data cache silently served stale coefficients.** The cache key omitted `fit_bases`,
  `sim_thetas`, `eval_times` and the ensemble grid size. Now keyed on a content hash, so a
  parameter change is a cache miss.
- **Mode search could run forever** — no iteration cap, and the stall test compared floats for exact
  equality. Added `mode_max_iterations` (default 50); progress is saved every iteration.
- **`get_ADMs` silently returned fewer rows than requested** when an ADM file was missing, which
  would misalign every ADM with the wrong LMK in the error propagation. Now raises.
- **`calc_type = 1` (scipy backend) crashed** on an argument-count mismatch.
- **Broadcasting bug in `calculate_coeffs_ensemble_scipy`**, and a `constant_background` variance
  that was 1-D where the rest of the code expected 2-D.

### Added

- **`how_to_run.md`** — a task-ordered guide written for occasional programmers: setup, run, plot,
  verify, then reference material. States expected output at every step.
- **`issues/`** — 20 defect reports, one per file, each with symptom, location, mechanism,
  reproduction and suggested fix, indexed by severity in `issues/README.md`.
- **`scripts/test_physics.py`** — 14 physics regression tests covering the frame rotation, agreement
  between all three calculation backends, coefficient ordering, the spherical harmonics, the C++
  Bessel evaluation, the likelihood, the measured-data round-trip, and the plotting fix.
- **`scripts/stage_adms.py`** to arrange the ADMs into the layout the reader expects, and
  **`scripts/analyse_run.py`** to report convergence and retrieved-vs-truth after a run.
- **`requirements.txt`** pinning the verified dependency set.
- **`external_artifacts/`** — the two run-time modules and the electron scattering amplitudes that
  ship with neither the repository nor PyPI, recovered from the cluster archive and vendored with
  provenance, so the repository is self-contained.
- **`max_iterations`** so a run stops deterministically instead of depending on a convergence test
  that needed ~120 000 steps to pass.
- A **"Fast run" section** in `analyze_results.ipynb` that plots a completed run without re-running
  the simulation.

### Changed

- **numpy and scipy API migrations**: `np.complex`/`np.int`/`np.float` → builtins, and
  `sph_harm` → `sph_harm_y`. The harmonic migration swaps *both* degree/order and the angle order,
  so it was verified numerically (agreement 1.1e-15 against analytic closed forms) rather than
  assumed — the naive unswapped call differs by 0.243.
- **All paths are repo-relative**, derived from `__file__`. The four hardcoded `/cds/` and `/reg/`
  cluster paths are gone.
- **The ensemble grid is named**: `ENSEMBLE_GRID_N` and `ENSEMBLE_GRID_SPAN` replace magic numbers.
- `README.md` and `CLAUDE.md` rewritten around a verified quick-start.

### Removed

- The committed `c_calc_extensions.so` — it was a Linux x86-64 binary that could not load on macOS.
  It is now built by `setup.sh` and gitignored.
- A broken macOS shortcut blob committed as a regular file, pointing at an external USB drive, that
  had never resolved for anyone cloning the repository.

### Verified

A 3000-step run (32 walkers, asymmetric NO₂, σ = 0.0962 ≙ the paper's SNR 100, q ∈ [0.5, 10] Å⁻¹)
recovers **all six parameters within 1σ of the truth the data was simulated from**, and the mode
search finds Θ\* to 5–6 significant figures. Bond-length resolution is 0.62–0.67 mÅ against the
paper's Table 1 value of ~0.5 mÅ. `tau` levels off near 71 rather than tracking chain length, which
it did while the centre-of-mass bug was present.

Two honest caveats: the *width* parameters are far more loosely determined than the means and are
skewed, as the paper's Figs. 6a and 8 predict; and `has_converged` still reports `False` because the
built-in criterion wants ~6800 steps, so the chain is *equilibrated* rather than formally converged.

### Known issues

Catalogued in [`issues/`](issues/). The one that can silently change published numbers is
[019](issues/019-prefactor-convention-differs-from-eq21.md), a coefficient normalisation that
differs from the paper's Eq. 21 by an L-dependent factor; it cancels for simulated data but not for
imported measured data. Symmetric-NO₂ (`experiment="2dof"`) does not run
([003](issues/003-2dof-symmetric-path-unwired.md)).

## [v0.0.1] — 2023-03-01

The state of the code at publication of Hegazy et al., *Communications Physics* **6**, 325 (2023).
Runs only on the original SLAC/LCLS cluster.

---

# Appendix — full engineering record

Everything below is the detailed record: every file recovered and every line changed, kept for
review and for porting to another branch.

## What had to be recovered: `external_artifacts/`

Two Python modules and the electron scattering amplitudes are needed at run time but ship with
neither the repo nor PyPI. They have been recovered from archived copies of the original cluster
environment and vendored into `external_artifacts/` (~316 KB total), so the repo is now
self-contained. `external_artifacts/README.md` holds the full provenance table.

| Vendored file | Recovered from | Needed by |
|---|---|---|
| `modules/fitting.py` | `baseTools.zip` → `cds/home/k/khegazy/baseTools/modules/fitting.py` | `fit_legendres_images`, called at `modules/density_extraction.py:2891` — **StoN error model only** |
| `modules/diffraction_simulation.py` | `simulations.zip` → `cds/home/k/khegazy/simulation/diffractionSimulation/modules/diffraction_simulation.py` | `diffraction_calculation`, called at `:2852`/`:2861` — **StoN error model only** |
| `scattering_amplitudes/3.7MeV/*_dcs.dat` | `cds/home/k/khegazy/BIGR/scattering_amplitudes/3.7MeV/` | `get_scattering_amplitudes`, `modules/NO2.py:646` — **always required** |

**Picking the right `fitting.py` matters.** Several copies exist in the archive. Only the one
vendored here defines `fit_legendres_images` with the `image_stds=` and `chiSq_fit=` keyword
arguments that `density_extraction.py:2891` actually passes; the others raise `TypeError`.

Because the two module imports are now lazy, deleting `external_artifacts/modules/` still leaves
you able to run the `constant_sigma`/`constant_background` error models, `mode_search.py`, and
every results/plotting path. The scattering amplitudes, by contrast, are required by all paths.

### Scattering amplitude file format

ELSEPA-style differential cross-section tables, parsed by **fixed column slices** (not
whitespace splitting) at `modules/NO2.py:646-664`:

- the first **31 lines are skipped** as header,
- `line[2:11]` → scattering angle in degrees,
- `line[39:50]` → differential cross section.

The reader converts to `q = 4π sin(θ/2)/λ` and returns `sqrt(dcs)` as a **cubic interpolator with
no extrapolation**, so your `dom` must lie inside the tabulated q range. Files are named
`<full element name>_dcs.dat`, from the `atom_info` table at `density_extraction.py:476-484` —
note it spells fluorine `flourine`. NO₂ needs `nitrogen_dcs.dat` and `oxygen_dcs.dat`.

---

## Every code change

### The most important fix: a centre-of-mass bug that corrupted every C coefficient

`rotate_to_principalI` and `rotate_to_principalI_ensemble` shifted to the centre of mass like this:

```python
molecules -= np.sum(molecules*self.mass[:,0], axis=-2, keepdims=True)
molecules /= np.sum(self.mass)
```

The centre of mass is `Σᵢmᵢrᵢ / Σᵢmᵢ`, but the division was applied to **`molecules`** rather than
to the mass-weighted sum. Algebraically that gives

```
r'' = (r − Σⱼmⱼrⱼ)/M  =  r/M − Σⱼmⱼrⱼ/M       instead of      r − Σⱼmⱼrⱼ/M
```

so **every coordinate was scaled by 1/M** as well as mis-centred. For NO₂, M = 46, and all pairwise
distances came out 46× too small — 0.029/0.048/0.023 Å where the geometry is 1.35/2.213/1.05 Å. A
rotation cannot change distances, which is what makes this unambiguous.

**The fix:**

```python
molecules = molecules \
    - np.sum(molecules*self.mass[:,0], axis=-2, keepdims=True)/np.sum(self.mass)
```

(assigning rather than `-=`/`/=` also stops it mutating the caller's array).

**Why it surfaced as an L-dependent anomaly.** With r ≈ 0.03 Å the spherical-Bessel argument q·ΔR
falls to ~0.02–0.24, and the C++ **upward** recursion is catastrophically unstable for l > x. So the
higher orders returned garbage: at a single geometry C₄₀₀ came out 7.6 and C₆₀₀ came out 8.4 × 10⁹
where the correct values are 1.2e-6 and 1.4e-9. This is the same instability the README warns about
for low q, triggered by the wrongly-small distances rather than by the q range.

**What it invalidated.** Everything measured before this fix, including several conclusions in the
`issues/` folder that are now retracted — most notably
[issues/016](issues/016-ensemble-quadrature-error-dominates-likelihood.md), which attributed a
rugged, non-convergent likelihood to the ensemble quadrature. With the fix, `logL` converges with the
ensemble grid to five significant figures from N = 19 upward (N = 11 agrees to 0.001 %).

**After the fix**, all of the following hold, and are checked by `scripts/test_physics.py`:

| check | result |
|---|---|
| rotation preserves pairwise distances | exact |
| \|C₄₀₀\| > \|C₆₀₀\| | 2.95 > 1.94 ✓ |
| coefficients fall with L | 7.07 > 2.95 > 1.94 ✓ |
| C₂₀₀/C₄₀₀ | 2.4 — matching paper Fig. 3c's ×2 scale factor |
| C++ / scipy-combination / scipy-Bessel backends | agree to 1.7e-10 |
| C₀₀₀ vs C₂₀₀ | 29.3 vs 11.0, same order (was ~3000× apart) |

**How it went unnoticed for so long:** every independent cross-check in the package was broken.
`compare_c_coeffs_scipy` (the built-in C++-vs-scipy test) raised `NameError`;
`calculate_coeffs_ensemble_scipy` had a broadcasting bug *and* ignored its weights argument;
`calc_type=1` raised `TypeError` on construction; `calc_type=2` raised two `NameError`s. All are now
fixed, and there is a regression suite.

### Environment-driven API migrations

| File | Line(s) | Before | After | Why |
|---|---|---|---|---|
| `modules/density_extraction.py` | 1428, 1476, 1612 | `np.complex(0,1)` | `1j` | removed in numpy 1.24 |
| " | 2815, 2851 | `np.int(...)` | `int(...)` | removed in numpy 1.24 |
| " | 1015 | `dtype=np.float` | `dtype=float` | removed in numpy 1.24 |
| " | 1432, 1485, 1617 | `sp.special.sph_harm(m, n, azim, polar)` | `sp.special.sph_harm_y(n, m, polar, azim)` | `sph_harm` removed in scipy 1.17 |
| `external_artifacts/modules/diffraction_simulation.py` | 90, 208, 415 | `np.complex(0,∓1)`, `dtype=np.complex` | `∓1j`, `dtype=complex` | same |
| " | 37, 419, 427 | `sph_harm(m, l, azim, polar)` | `sph_harm_y(l, m, polar, azim)` | same; line 37 also dropped a `dtype=np.complex64` kwarg `sph_harm_y` does not accept, so the result is cast instead |
| `modules/plot_functions.py` | 54 | `fig.gca(projection='3d')` | `fig.add_subplot(projection='3d')` | removed in matplotlib 3.6 |

#### The spherical-harmonic migration was verified, not assumed

`sph_harm_y` swaps **both** the degree/order and the angle order relative to `sph_harm`, so
getting it wrong would silently corrupt the physics instead of raising. Before touching any call
site, the mapping

```
sph_harm(m, n, azim, polar)  ==  sph_harm_y(n, m, polar, azim)
```

was checked in the same interpreter (scipy 1.15.3 has both) and agreed to **1.1 × 10⁻¹⁵** over
all l = 0…6, m = −l…l. It was independently confirmed against the analytic closed forms
Y₁⁰ = √(3/4π)·cos θ, Y₁¹ = −√(3/8π)·sin θ·e^{iφ} and Y₂⁰ = √(5/16π)(3cos²θ − 1), and re-checked
under the exact broadcast shapes of all three call sites (agreement 5.6 × 10⁻¹⁶). The naive
*unswapped* call differs by 0.243, confirming the swap is load-bearing.

If you ever move to scipy ≥ 1.17 and want to re-verify, the check is worth repeating.

### Making the code runnable off the cluster

**`modules/density_extraction.py:21-26`** — `from diffraction_simulation import …` and
`from fitting import …` were unconditional module-level imports (the surrounding
`os.path.exists("/cds/…")` guards only appended to `sys.path`, they did not make the imports
optional), so merely importing `density_extraction` failed. Replaced with an
`external_artifacts/modules` entry derived from `__file__`, and **both imports moved inside
`simulate_error_StoN`**, the only function that uses them. That is what lets the
`constant_sigma` path, `mode_search.py`, and all plotting run without them.

Note it uses `os.path.realpath`, not `abspath`: this file is normally reached through the
`NO2/modules` symlink, so `abspath` resolves the repo root to `NO2/` and misses
`external_artifacts` entirely. (That was a real bug in the first attempt.)

**`modules/density_extraction.py`, `run_mcmc`** — added a hard iteration cap after the
checkpoint save:

```python
if self.sampler.iteration >= self.data_params.get("max_iterations", np.inf):
    print("INFO: reached max_iterations ({}), stopping. has_converged = {}".format(
        self.data_params["max_iterations"], self.has_converged))
    break
```

Without it the loop is effectively unbounded: exit requires `iteration > 100·τ` **and** τ stable
to 1% **and** `iteration > min_acTime_steps·τ`, while the only pre-existing escape (`if
np.amax(tau) > 500 and iteration/np.amax(tau) > 15`) never fires for a chain whose τ plateaus
around 40.

**`external_artifacts/modules/fitting.py`, `normal_eqn_vects`** — two bugs in the
singular-matrix guard, both of which stopped the StoN path outright:

1. It tested `np.linalg.det(overlap) == 0.0` exactly. A rank-deficient matrix generally gives a
   tiny denormal determinant rather than a true zero, so the test missed and `np.linalg.inv`
   raised `LinAlgError`. This fires at small radii, where a ring has fewer pixels than the number
   of Legendre orders being fit — radius 0 is a single pixel against 4 orders. Replaced with a
   conditioning test (`np.linalg.cond > 1/eps`) plus a non-finite check.
2. The NaN return used `X.shape[0]` (the batch size) where it needed `X.shape[1]` (the number of
   Legendre orders), so degenerate radii returned wrongly-shaped arrays and the caller's
   `np.concatenate` failed. Returning NaN there is the original intent; those radii sit at
   q below `fit_range` and are masked out downstream (verified: the only NaN in the final
   coefficients is at q = 0).

**`modules/NO2.py`** — the ensemble grid size was the hardcoded literal `N = 19` inside
`molecule_ensemble_generator`. Promoted to a documented module-level constant
`ENSEMBLE_GRID_N` (now `11`) because it is by far the biggest lever on runtime; see
`how_to_run.md` §7.

**`NO2/mode_search.py`** — added explicit `import argparse` and `import time`. They previously
resolved only by accident, because `from modules.NO2 import *` happened to re-export
`modules/NO2.py`'s own imports. Adding an `__all__` to that module, or reordering the imports,
would have broken the script.

**`NO2/build_posterior.py:119-120` and `NO2/mode_search.py:134`** — `lmk_arr` was assigned
twice in `build_posterior.py`, so the second assignment silently won and disagreed with
`mode_search.py`'s list. Since `ADM_params` feeds `get_fileName`, that made `mode_search.py`
look in a folder `build_posterior.py` never wrote to. Both lists now read `[[30, 100]]` and the
sweep variant is commented out.

**`NO2/parameters.py`** — all four cluster paths replaced with repo-relative ones derived from
`__file__` (so they resolve from any working directory, which the notebook needs):

```python
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../BIGR/NO2
_REPO = os.path.dirname(_HERE)                        # .../BIGR
_ART  = os.path.join(_REPO, "external_artifacts")
```

| Key | Was | Now |
|---|---|---|
| `scat_amps_dir` | `/cds/home/k/khegazy/simulation/scatteringAmplitudes/3.7MeV/` | `_ART/scattering_amplitudes/3.7MeV` |
| `output_dir` | `/cds/group/ued/scratch/khegazy/TeDDE/` | `_REPO/output` |
| `save_sim_data` | `/cds/group/ued/scratch/khegazy/TeDDE/saved_simulations/` | `_REPO/output/saved_simulations` |
| `ADM_params["folder"]` | `/reg/data/ana15/ued/scratch/khegazy/TeDDE/axis_distributions` | `_REPO/NO2_properties_simulation/axis_dist/bigr_layout` |
| `init_geo_xyz` | `"XYZ/NO2_symbreak.xyz"` (cwd-relative) | absolute, via `_HERE` |

### `NO2/analyze_results.ipynb`

- Cell 0: dropped `calc_ensemble_dists, calc_dists` from the `modules.density_extraction`
  import — **those names do not exist**; the class has *methods* `calculate_dists`
  (`density_extraction.py:1298`) and `calculate_ensemble_dists` (`:1327`). This made the very
  first cell raise `ImportError`.
- Cell 0: `from IPython.core.display import display, HTML` → `from IPython.display import …`
  (removed in modern IPython), and `from diffraction_simulation import diffraction_calculation`
  wrapped in `try/except ImportError` so the notebook opens without the vendored module.
- Notebook metadata pinned a kernel named `emcee_env`, which does not exist here, so
  `nbconvert --execute` failed with `NoSuchKernel`. Repointed at a `bigr` kernelspec:
  `.venv/bin/python -m ipykernel install --user --name bigr --display-name "BIGR (.venv)"`.
- **Added a "Fast run" section** (6 new cells after cell 2) that plots the short run using the
  `results_only=True` construction already used further down — it reads only the backend HDF5,
  skipping ADMs, scattering amplitudes and simulation. It prints the truth/median/σ^Θ/Θ\* table
  and writes `fast_run_corner.png` and `fast_run_chains.png`.
  The corner plot is drawn from the **un-thinned** post-burn-in chain (`thin=False`, ~21 000
  correlated samples) because the τ-thinned chain has only ~160 independent samples — too few for
  contours. σ^Θ is quoted from the thinned, independent samples.
- Outputs stored in the original sweep cells were **preserved**; a markdown cell now states
  explicitly that they come from the authors' cluster runs and not from anything in this
  checkout.

### Repository housekeeping

- Added `.gitignore` (there was none), anchored with leading `/` so that pre-existing figure
  directories such as `NO2_properties_simulation/plots/` stay tracked.
- **Untracked `cpp_extensions/lib/c_calc_extensions.{so,o}`.** They were committed as Linux
  x86-64 binaries; rebuilding for macOS would otherwise commit arm64 binaries and break Linux
  users. Everyone builds their own now.
- Added `scripts/stage_adms.py` and `external_artifacts/README.md`.

### `setup.sh` — rewritten

The original could not run at all, in six separate ways (see
[issues/012](issues/012-setup-sh-broken.md)): an unterminated double quote on the last line made it
unparseable; two `ln -s` calls were guarded by an undefined `$FILE` and so silently never ran; one
`wget` host was a typo (`githubi.com`); both `wget` URLs were GitHub `/blob/` pages that serve HTML
rather than Python; and `mkdir output` lacked `-p`, so any re-run failed.

It has been rewritten: `set -euo pipefail`, paths derived from `${BASH_SOURCE[0]}` rather than a
chain of bare `cd`s, idempotent `mkdir -p`/`ln -sfn`, honest reporting of whether the C++ build
succeeded, and the `wget` block deleted (both modules are now vendored in `external_artifacts/`, and
re-downloading them from the pinned upstream commits would reintroduce the numpy/scipy
incompatibilities fixed here). It also stages the ADMs and prints the remaining steps.

**`bash setup.sh` is now the recommended way to set up.**

---
