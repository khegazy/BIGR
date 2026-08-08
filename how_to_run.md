# How to run the BIGR NO₂ analysis

This is a step-by-step record of getting the NO₂ example running from a clean checkout on a
machine that is **not** the original SLAC/LCLS cluster, including every file that had to be
recovered and every line of code that had to change. It also specifies the HDF5 layouts, so you
can feed your own measured data into `NO2/analyze_results.ipynb`.

The method is described in `BIGR_paper.pdf` — Hegazy et al., *Applying Bayesian inference and
deterministic anisotropy to retrieve the molecular structure |Ψ(R)|² distribution from gas-phase
diffraction experiments*, **Communications Physics 6, 325 (2023)**,
[doi:10.1038/s42005-023-01420-9](https://doi.org/10.1038/s42005-023-01420-9).

**Contents**
1. [TL;DR](#1-tldr)
2. [Environment](#2-environment)
3. [What had to be recovered: `external_artifacts/`](#3-what-had-to-be-recovered-external_artifacts)
4. [Symmetric vs. asymmetric NO₂ — pick the right degrees of freedom](#4-symmetric-vs-asymmetric-no2--pick-the-right-degrees-of-freedom)
5. [Staging the ADMs](#5-staging-the-adms)
6. [Every code change](#6-every-code-change)
7. [Parameter reference and what each knob costs](#7-parameter-reference-and-what-each-knob-costs)
8. [Choosing an error model](#8-choosing-an-error-model)
9. [HDF5 file formats](#9-hdf5-file-formats)
10. [Bringing your own measured data](#10-bringing-your-own-measured-data)
11. [Gotchas](#11-gotchas)
12. [Known remaining issues](#12-known-remaining-issues)

---

## 1. TL;DR

```bash
cd /path/to/BIGR

# 1. environment (uv-managed venv; this venv has no pip, use uv)
uv pip install --python .venv/bin/python \
    numpy scipy matplotlib h5py emcee corner tqdm ipykernel nbconvert

# 2. symlinks + output dirs that setup.sh was supposed to make (do NOT run setup.sh, see below)
ln -s ../modules NO2/modules
ln -s ../cpp_extensions NO2/cpp_extensions
mkdir -p NO2/plots output/saved_simulations

# 3. build the C++ extension for THIS platform (the committed .so was Linux x86-64)
cd cpp_extensions/lib && make clean && make && file c_calc_extensions.so   # expect Mach-O arm64
cd ../..

# 4. reshape the in-repo ADMs into the layout get_ADMs expects
.venv/bin/python scripts/stage_adms.py

# 5. run the analysis (from inside NO2/ -- the .so path is resolved relative to the cwd)
cd NO2
MPLBACKEND=Agg ../.venv/bin/python build_posterior.py     # ~10 min: posterior P(Theta|C)
MPLBACKEND=Agg ../.venv/bin/python mode_search.py         # mode Theta*

# 6. plots -- open the notebook and run the "Fast run" section (cells 3-8) only
cd ..
.venv/bin/python -m ipykernel install --user --name bigr --display-name "BIGR (.venv)"
.venv/bin/jupyter lab NO2/analyze_results.ipynb    # select the "BIGR (.venv)" kernel
```

**Do not execute the whole notebook.** Only the first nine cells belong to the fast run;
everything after them is the paper's parameter sweeps and needs its own completed MCMC run per
configuration (hours to days). To regenerate the fast-run plots headless, execute just that
prefix:

```bash
.venv/bin/python - <<'EOF'
import json, copy
nb = json.load(open("NO2/analyze_results.ipynb"))
sub = copy.deepcopy(nb); sub["cells"] = nb["cells"][:9]
json.dump(sub, open("NO2/_fast.ipynb", "w"), indent=1)
EOF
cd NO2 && MPLBACKEND=Agg ../.venv/bin/python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=bigr --ExecutePreprocessor.timeout=1800 _fast.ipynb
rm _fast.ipynb            # plots are already written to NO2/plots/...
```

The trimmed notebook must live in `NO2/` because the kernel's working directory follows the
notebook's location, and `from parameters import *` plus the cwd-relative `.so` load both
require it.

Do **not** pass `--multiProc_ind` — see [Gotchas](#11-gotchas).

Outputs land in:

| Path | Contents |
|---|---|
| `output/<molecule>/<experiment>/sim/<density_model>/<T>K_<I>TW_<FWHM>fs/results_*.h5` | MCMC backend (the posterior) |
| `…/mode_search_results_*.h5` | mode Θ\* |
| `output/saved_simulations/<same relative path>/results_*.h5` | cached simulated C coefficients |
| `NO2/plots/…` | corner plot, walker trajectories, setup diagnostics |

### What a 2-hour run actually produces

Asymmetric NO₂, PDF model, `constant_sigma = 0.163`, q ∈ [0.5, 5] Å⁻¹, ADMs at 1 K, shipped
`ENSEMBLE_GRID_N = 19`, 32 walkers × **2400 steps (2 h wall clock)**, walkers initialised *at* the
ground truth:

| Θ | truth | posterior median | σ^Θ | mode Θ\* | (med−truth)/σ^Θ | unit |
|---|---|---|---|---|---|---|
| ⟨NO⁽¹⁾⟩ | 1.35000 | 1.89457 | 0.34396 | 1.4756 | +1.58 | Å |
| σ(NO⁽¹⁾) | 0.03000 | 0.13493 | 0.10416 | 0.0756 | +1.01 | Å |
| ⟨NO⁽²⁾⟩ | 1.05000 | 1.44789 | 0.28488 | 1.2698 | +1.40 | Å |
| σ(NO⁽²⁾) | 0.02000 | 0.10586 | 0.07898 | 0.0666 | +1.09 | Å |
| ⟨∠ONO⟩ | 2.34000 | 1.87766 | 0.46421 | 1.8266 | −1.00 | rad |
| σ(∠ONO) | 0.01000 | 0.19357 | 0.13571 | 0.1700 | +1.35 | rad |

acceptance 0.257 · τ ≈ 134 · `has_converged = False` · 384 independent samples

**Read this carefully — the retrieval does not recover the truth at these settings, and that is not
a code fault.** The evidence separates cleanly:

*The forward model and likelihood are correct.*
- `logL(truth) = 0` exactly — the model reproduces its own simulated data.
- **Nothing beat truth** among 2400 × 32 = 76 800 samples (best −0.126). Truth is the global maximum.
- The spherical-Bessel C++ backend agrees with scipy to 1e-14 (`plots/check_jl*.png`).
- Acceptance 0.257 is healthy for emcee, and the chain is stationary from step ~600 onward.

*But the configuration carries too little information.* At σ = 0.163 and q ≤ 5 Å⁻¹, broadening the
ensemble costs almost nothing in likelihood while the flat prior on the widths (`[0, 0.5]`) gives the
broad region far more volume. Posterior mass ∝ likelihood × volume, so mass collects away from the
peak: the widths settle 5–20× high and σ(∠ONO) visibly piles up against its prior ceiling in
`fast_run_chains.png`. See
[issues/018](issues/018-posterior-prior-dominated-at-low-information.md).

The mode search does not rescue it: Θ\* has `logL = −8.73`, worse than the posterior median's −0.90,
because `weight_avg_search` computes a weighted centroid rather than maximising — see
[issues/017](issues/017-mode-search-returns-worse-than-median.md).

**The single most useful lesson:** initialising at the truth and reading off agreement after a short
run proves nothing. A 1000-step run at these settings looked excellent (all six parameters within
0.25σ) purely because the walkers had not yet left their starting ball. Only the longer run revealed
the posterior's real shape.

Diagnostics to check on any run (`python scripts/analyse_run.py`):

- **acceptance fraction** — want ~0.2–0.5.
- **τ / steps** — must fall below 0.01 for `conv1` (`iteration > 100·τ`). A constant ~0.11 across
  batches means emcee's estimator is saturating against chain length.
- **(median − truth)/σ^Θ** — the correctness check for simulated data.
- **width parameters at their prior bound** — if σ(·) accumulates near `sig_max = 0.5`, the answer is
  set by the prior, not the data.

To reach the paper's regime you need a properly calibrated error model (`StoN`, currently unusable —
[issues/001](issues/001-ston-signal-to-noise-unusable.md)), q ≤ 10–20 Å⁻¹, and more walkers.

Plots in `NO2/plots/NO2_symbreak/3dof/sim/PDF/1K_10TW_100fs/`: `fast_run_corner.png` and
`fast_run_chains.png`.

## 2. Environment

The venv at `.venv/` is Python 3.10.20, created by `uv`, and therefore **has no `pip`** — use
`uv pip install --python .venv/bin/python …`. Resolved versions:

| Package | Version |
|---|---|
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| matplotlib | 3.10.9 |
| h5py | 3.16.0 |
| emcee | 3.1.6 |
| corner | 2.3.0 |

These are far newer than the floors in `README.md` (numpy ≥ 1.21.4, scipy ≥ 1.6.2), which is why
the API migrations in [§6](#6-every-code-change) were needed. `tqdm` is required because
`run_mcmc` calls `sampler.run_mcmc(..., progress=True)`. Always set `MPLBACKEND=Agg`; every plot
is written with `savefig` and an interactive backend just risks GUI/threading problems.

**scipy 1.15.3 is a deliberate sweet spot.** It is the last release supporting Python 3.10, and
it is the only window in which *both* the removed `sph_harm` and its replacement `sph_harm_y`
exist. That let the spherical-harmonic migration be verified numerically rather than guessed —
see [§6](#6-every-code-change).

---

## 3. What had to be recovered: `external_artifacts/`

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

## 4. Symmetric vs. asymmetric NO₂ — pick the right degrees of freedom

**This is the easiest thing to get silently wrong.** The number of structural degrees of freedom
must match the geometry you intend to retrieve, and the repo contains both geometries.

| | Symmetric NO₂ | Asymmetric ("symmetry-broken") NO₂ |
|---|---|---|
| XYZ file | `NO2/XYZ/NO2.xyz` | `NO2/XYZ/NO2_symbreak.xyz` |
| Bond lengths | both N–O = **1.1934 Å** | N–O = **1.35 Å** and **1.05 Å** |
| ∠ONO | 2.337 rad (133.9°) | 2.337 rad |
| Θ | (d, σ_d, ∠, σ_∠) → **one shared bond DOF** | (d₁, σ_d₁, d₂, σ_d₂, ∠, σ_∠) → **each bond its own DOF** |
| `molecule` | `"NO2"` | `"NO2_symbreak"` |
| `experiment` | `"2dof"` | `"3dof"` |
| Cartesian builder | `theta_to_cartesian_2dof` (`modules/NO2.py:196`) reuses `theta[:,0]` for **both** oxygens (lines 213-216) | `theta_to_cartesian_single` / `_ensemble` (`:146`, `:171`) use independent `theta[:,0]`, `theta[:,1]` |
| ndim (PDF / delta) | 4 / 2 | **6 / 3** |
| Works today? | **No** — see below | **Yes** |

`get_parameters()` already couples these correctly: `molecule` selects the XYZ file
(`parameters.py:86-89`) and `experiment` selects `sim_thetas` (`:123-144`). If you change one,
change the other.

**The 2dof path is incomplete.** `theta_to_cartesian_2dof` exists but **no generator calls it** —
`single_molecule_generator` and `molecule_ensemble_generator` both build 3dof geometries
(`modules/NO2.py:342`, `:411`). So setting `experiment="2dof"` pairs the 2dof *log priors* with
3dof *geometry generation* and raises an `IndexError`. To finish it you would add
`single_molecule_generator_2dof` / `molecule_ensemble_generator_2dof` that call
`theta_to_cartesian_2dof`, and select them in `NO2/build_posterior.py:50-68` alongside the
existing `log_prior_2dof_*` choice.

**This run uses the asymmetric case**, which is also the paper's headline simulated result
(Table 1: ⟨NO⁽¹⁾⟩ = 1.3500 Å, ⟨NO⁽²⁾⟩ = 1.0500 Å, ∠ONO = 2.34 rad).

---

## 5. Staging the ADMs

The axis distribution moments (ADMs, paper Eq. 1) live in
`NO2_properties_simulation/axis_dist/`. They are already `.npy`, but in a different directory
layout than `modules/NO2.py:get_ADMs` reads, so `scripts/stage_adms.py` reshapes the tree. No
numbers are changed and `modules/NO2.py` is left untouched:

| | Source | What `get_ADMs` expects (`NO2.py:536-546`) |
|---|---|---|
| bases | `axis_dist/A/temp-{T}K/A-Mat_NO2{word}K I=10TW tau=100fs D{L}{K}.npy` | `<folder>/NO2/ADMs/temp-{T}K/{I}TW_{FWHM}fs/` matching glob `A*.npy` |
| times | `axis_dist/times/temp_{T}K.npy` | a literal `times.npy` in that same directory |

The basis **filenames are preserved deliberately**: `get_ADMs` parses the angular momentum
indices positionally from the end of the path, `L = int(fl[-6])` and `K = int(fl[-5])`
(`NO2.py:558-559`), so `… D62.npy` yields L=6, K=2. Per `axis_dist/README.txt` the `D{K}{S}`
suffix *is* the ADM index ("the D22 file is the ADM A²₀₂"), matching the code's `[L, 0, K]`
convention. Run it with:

```bash
.venv/bin/python scripts/stage_adms.py            # all temperatures
.venv/bin/python scripts/stage_adms.py --temps 1  # just 1 K
```

### Two hard limits of this ADM set

1. **Temperatures are 1, 10, 20, 30 and 300 K only** — never the `100` that `parameters.py`
   originally requested. `ADM_params["temperature"]` must be one of those five.
2. **The time axis spans only −0.20 → 40.30 ps** (502 points). The original
   `eval_times = linspace(37.5, 41.5, 100)` runs past the end, so the cubic `interp1d` at
   `NO2.py:580` raises. It is now `linspace(16.0, 20.0, 25)`.

The window is not arbitrary. Measuring the mean-subtracted ADM amplitude in 4 ps windows shows
the rotational revival — the anisotropy the whole method depends on — peaks near **18 ps**:

```
temp-1K, |A20 − mean| rms per 4 ps window
  [0-4] 0.0097   [4-8] 0.0092   [8-12] 0.0087  [12-16] 0.0043
 [16-20] 0.0217  [20-24] 0.0100 [24-28] 0.0066 [28-32] 0.0109
 [32-36] 0.0073  [36-40] 0.0190
```

Temperature matters even more for the *higher-order* moments, which is what constrains the
molecular-frame angles (paper Fig. 6c-d):

| ADM | 1 K | 30 K | ratio |
|---|---|---|---|
| A²₀₀ (`D20`) peak-to-peak | 0.0689 | 0.0203 | 3.4× |
| A⁴₀₀ (`D40`) | 0.0609 | 0.0059 | 10× |
| A⁶₀₀ (`D60`) | 0.0662 | 0.0017 | **39×** |

Hence `temperature: 1`.

---

## 6. Every code change

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
[§7](#7-parameter-reference-and-what-each-knob-costs).

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

### `setup.sh` — do not run it

Left as-is (it documents original intent) but it is broken in six ways -- see [issues/012](issues/012-setup-sh-broken.md):
- line 37 has an **unterminated double quote** → shell syntax error;
- lines 15 and 25 test `$FILE`, which is never defined (the variables are `DATA_PARAMS_FILE`
  etc.), so those `ln -s` silently never run;
- line 31's host is `githubi.com`, a typo;
- lines 31-32 use GitHub `/blob/` URLs, which download HTML pages rather than Python files;
- line 8 is `mkdir output` without `-p`, so it fails on any re-run.

Step 2 of the [TL;DR](#1-tldr) does what it was meant to do.

---

## 7. Parameter reference and what each knob costs

### On the ensemble grid size

An earlier version of this document warned that the ensemble quadrature dominated the likelihood and
that `ENSEMBLE_GRID_N` must never be lowered. **That was wrong** — it was an artifact of the
centre-of-mass bug (see below), which shrank every molecule by its total mass and pushed the C++
Bessel recursion into its unstable regime. With that fixed, `logL` converges with the grid to five
significant figures from N = 19 upward, and even N = 11 agrees to 0.001 %. See
[issues/016](issues/016-ensemble-quadrature-error-dominates-likelihood.md), now retracted.

So `ENSEMBLE_GRID_N` is a straightforward speed/accuracy knob after all: cost scales as N³ and
accuracy saturates quickly. 19 is the shipped value and is comfortably converged.

### The override trap

`get_parameters()` sets `multiprocessing`, `Nwalkers` and `run_limit` in the dict literal and
then **overwrites all three** in the `density_model` branch at `parameters.py:95-102`. Editing
the literal alone does nothing. Edit the branch — for `density_model="PDF"` that is the `else`
clause.

### Fast settings actually used, versus the paper

| Parameter | Paper / original | Here | Effect |
|---|---|---|---|
| `ENSEMBLE_GRID_N` (`modules/NO2.py`) | 19 | **19** | **The dominant cost** (∝ N³, full outer product over 3 DOF): 19³ = 6859 geometries/walker → ~4.9 s per MCMC step at 118 q points. **Do not reduce it** — see the warning below. |
| `ENSEMBLE_GRID_SPAN` (`modules/NO2.py`) | 7 (was hardcoded) | 7 | Grid half-width in σ. **Do not reduce it either**; ±3σ is 200× less accurate at fixed N because the integrand oscillates. |
| `multiprocessing` | 10 | **0** | Must be 0 or 1 on macOS. `calculate_c_ensemble_multiProc` pickles a bound method whose `self.spherical_j` is a closure created in `setup_calculations:2500`; the spawn start method cannot pickle it. 0 keeps the fast in-process C++ path. |
| `Nwalkers` | 100 | 32 | Cost and memory are linear in this. Must be ≥ 2·ndim = 12. |
| `run_limit` | 100 | 100 | Batch size between checkpoints/convergence checks — **not** an iteration cap. |
| `min_acTime_steps` | 3000 | 5 | Removes the `iteration > 3000·τ` gate, which alone would need ~120 000 steps. |
| `max_iterations` | — (new) | 1000 | Deterministic stop. |
| `fit_range` | `[0.5, 10]` | `[0.5, 10]` | With `q_per_pix` doubled this gives `dom` = 118 points instead of the paper's 236. |
| `q_per_pix` | `3.5/83` | `2*3.5/83` | Halves `dom` again. |
| `calc_type` | 0 | 0 | C++ backend. |
| `simulate_error` | `("StoN", (100, [0.5,4]))` | `("constant_sigma", 0.05)` | See [§8](#8-choosing-an-error-model). |
| `ADM_params["temperature"]` | 100 | 1 | Only 1/10/20/30/300 K exist; cold gives far larger high-order ADMs. |
| `ADM_params["eval_times"]` | `linspace(37.5,41.5,100)` | `linspace(16.0,20.0,25)` | Must stay inside −0.2…40.3 ps and should bracket the 18 ps revival. |
| `mode_tolerance` | 1e-4 | 0.01 | Must be met 3 consecutive times. |
| `N_mode_samples` | 50 | 25 | Posterior samples seeding the mode search. |

To reproduce the paper's numbers, restore the left-hand column and expect hours to days per
configuration rather than minutes.

### Choosing σ for `constant_sigma`

σ is the standard deviation assigned to **every** C_lmk(q) point
(`density_extraction.py:1044`), and the likelihood is `−0.5·(data − calc)²/σ²`. So it sets the
resolution directly.

Calibrate it against the paper's convention, which defines SNR as the geometric mean of
C₀₀₀(q)/σ₀₀₀(q) over 0.5 < q < 4 Å⁻¹ (paper p. 16). Measured here, the geometric mean of |C₀₀₀| over
that window is **9.62**, so:

| paper SNR | σ | resulting S/N on the fitted L≥2 coefficients |
|---|---|---|
| 25 | 0.385 | 29, 3, 11, 3, 0.1, 6 |
| **100** (paper's standard) | **0.0962** | 115, 12, 44, 12, 0.5, 24 |
| 400 | 0.0241 | 459, 46, 176, 48, 2, 98 |

`σ = 0.0962` is used here, matching the paper's standard configuration. It yields conditional
resolutions of **0.63 mÅ** on ⟨NO⁽¹⁾⟩ and **1.1 mrad** on ⟨∠ONO⟩, against the paper's Table 1 values
of 0.5 mÅ and 1 mrad — good agreement.

Note `constant_sigma` still cannot reproduce the paper's *shape* of error bars, which are
per-coefficient and per-q. For that you need `StoN`
([issues/001](issues/001-ston-signal-to-noise-unusable.md)) or `constant_background`, which at least
gives the physically correct q dependence.

**Match `init_thetas_std_scale` to the posterior width.** It multiplies element-wise, so the walker
spread is `scale × init_thetas`. If the spread is much wider than the posterior almost every proposal
is rejected and parameters freeze (τ comes back as `nan`); much narrower and the chain spends
thousands of steps merely expanding. At σ = 0.0962 the tightest relative width is ~5e-4, which is the
value used.

### Mode search cost

`weight_avg_search` evaluates a grid of `len(mode_std_grid)**ndim` θ per iteration — with the
default 3-point grid and ndim = 6 that is **3⁶ = 729** likelihood evaluations per loop. Keep
`mode_std_grid` at length 3; a 5-point grid is 5⁶ = 15 625. `get_cs_fxn` batches θ in chunks of
`stride = 200` (`modules/mode_search.py:41`) — lower it if you run out of memory. There is a
hardcoded 1500-sample random-search fallback at `:333` if the grid search stalls.

---

## 8. Choosing an error model

There are four, and after the centre-of-mass fix (§6) three of them work. Pick by how faithful you
need the noise to be.

| model | error bars | status |
|---|---|---|
| **`("StoN", (SNR, q_range))`** | per-coefficient **and** per-q, from Poisson noise on simulated diffraction images propagated through the ADM fit | **works — use this for publication-grade results.** Needs the vendored modules and the ADMs |
| `("constant_background", σ)` | per-q: σ(q) ∝ 1/atomic_scattering(q), so the error grows with q; normalised on C₂₀₀ | works. The best approximation with no ADM dependency |
| `("constant_sigma", σ)` | one σ for every coefficient and every q | works, crudest |
| `("data", (order, Wn))` | taken from a real measurement | **broken** — [issues/010c](issues/010-dead-and-broken-code-paths.md) |

### `StoN` — the paper's model, and it works

At the paper's standard SNR = 100 with 25 `eval_times`, measured per-LMK S/N:

| LMK | S/N |
|---|---|
| [2 0 0] | **144.9** |
| [4 0 0] | **28.7** |
| [6 0 0] | **16.1** |
| [2 0 2] | 2.2 |
| [4 0 2] | 2.2 |
| [4 0 4] | 0.006 |

The k = 0 coefficients are well measured. The much lower S/N on k ≠ 0 is **expected physics**, not a
defect: those terms are constrained by the azimuthal ADMs A^L_{0K} with K ≠ 0, which are genuinely
smaller than the K = 0 moments.

An earlier version of this document reported StoN as unusable, with S/N ≈ 0.05. That measurement was
taken before the centre-of-mass fix, when the C coefficients were ~500× too small and partly
Bessel-instability noise — the error propagation was correct all along; the signal it was compared
against was not. See [issues/001](issues/001-ston-signal-to-noise-unusable.md).

Worth knowing about the mechanism either way: `simulate_error_StoN` converts lab-frame errors into
molecular-frame C_lmk errors via `inv(Aᵀ W A)` with the **mean-subtracted** ADM matrix `A`
(`density_extraction.py:2946-2953`). So the variance scales as 1/A² — stronger alignment helps
quadratically — and only as 1/N_times, meaning adding time points is an inefficient lever (S/N grows
as √N_times).

### `constant_sigma` — what the runs in this document use

Simple and ADM-free, which makes it a good default while you are getting the pipeline working, but its
error bars are uniform in both q and LMK, so it cannot reproduce the *shape* of real noise. Calibrate
it as described in §7.

### `constant_background` — the middle option

σ(q) ∝ 1/atomic_scattering(q), which is the physically right shape for a detector-level background
expressed in sms units — measured 7.3× growth across [0.5, 5] Å⁻¹ — and it is normalised on C₂₀₀, a
coefficient that is actually in `fit_bases`, rather than on C₀₀₀. It still applies the same σ(q) to
every LMK, so it is an approximation to StoN, not a substitute. **Note the parameter is 1/(C₂₀₀ S/N),
not the S/N**: 0.01 gives C₂₀₀ S/N ≈ 108. The docstring had this inverted and is now corrected.

### The retrieval is verified against known truth

Independently of the noise model:

- `logL(truth) = 0` **exactly** — the simulated data is the noiseless forward model at `sim_thetas`
  (no noise is ever added to the coefficients; the commented-out lines at
  `density_extraction.py:1107-1110` would have done it), so this is an identity and confirms
  self-consistency rather than goodness of fit.
- All 12 one-sided parameter perturbations lower `logL`, so truth is a local maximum.
- The C++, scipy-combination and scipy-Bessel backends agree to ~1e-10.
- Retrieval recovers all six Θ to within 0.1σ of truth — see §1.

All of these are asserted by `scripts/test_physics.py`.

## 9. HDF5 file formats

All files are flat: **datasets live at the root group**, no subgroups.

### 9.1 MCMC backend / results — `output/<folder>/results_*.h5`

Written by `save_emcee_backend` (`density_extraction.py:1934`) with mode `"w"`, i.e. fully
rewritten on every checkpoint. Read back by `load_emcee_backend` (`:1890`).

| Dataset | Shape | dtype | Notes |
|---|---|---|---|
| `chain` | `(n_steps, nwalkers, ndim)` | float64 | the posterior samples |
| `log_prob` | `(n_steps, nwalkers)` | float64 | log joint probability |
| `nwalkers` | scalar | int64 | |
| `ndim` | scalar | int64 | 6 for PDF/3dof, 3 for delta/3dof |
| `accepted` | `(nwalkers,)` | float64 | emcee acceptance counts |
| `has_converged` | scalar | bool | |
| `tau_convergence` | `(n_batches, ndim)` | float64 | τ history, one row per `run_limit` batch |
| `autocorr_times` | `(ndim,)` | float64 | latest τ. **Not** read back by `load_emcee_backend` |
| `filtered_chain` | `(n_thin, nwalkers, ndim)` | float64 | thinned chain, or the literal `np.array([False])` when `iteration < 4·max(τ)` |

`get_mcmc_results()` returns `chain` discarded by `3·max(τ)` and thinned by `max(τ)`, reshaped to
`(n_samples, ndim)`.

### 9.2 Mode search — `output/<folder>/mode_search_results_*.h5`

Written by `save_mode_search` (`modules/mode_search.py:153`), mode `"w"`.

| Dataset | Shape | Notes |
|---|---|---|
| `ths_mean` | `(ndim,)` | **Θ\*, the mode** — the only one the notebook reads |
| `ths_var`, `ths_std` | `(ndim,)` | spread of the weighted sample |
| `ths_mean_history`, `ths_std_history` | `(n_iters, ndim)` | per-iteration history |
| `ths_sampled` | `(ndim, ≤10000)` | θ actually evaluated |
| `log_prob_sampled`, `chiSq_sampled` | `(≤10000,)` | |

The filename is the backend's with `mode_search_` prefixed onto the **basename** (same
directory). That surgery is done in three separate places — `modules/mode_search.py:220-223` and
two notebook cells.

### 9.3 Simulated-coefficient cache — `output/saved_simulations/<same folder>/results_*.h5`

Written by `save_simulated_data` (`:1152`), read by `load_simulated_data` (`:1183`), which
returns `False` (and simulation proceeds) if the file is absent. Same relative path and
basename as the backend, just under `save_sim_data`.

| Dataset | Shape | Notes |
|---|---|---|
| `input_data_coeffs` | `(n_lmk, n_dom)` | pre-pruning, on the full `dom` |
| `input_data_coeffs_var` | `(n_lmk, n_dom)` | |
| `experimental_var` | `(n_lmk, n_dom)` | only written when not `None`; the loader sets it to `None` when absent |

A successful load forces `data_params["isMS"] = True`.

---

## 10. Bringing your own measured data

Set `data_params["data_fileName"]` to a **complete path to your file**. There is no name
construction and `output_dir` is not involved. Its presence is what switches
`get_data` (`:851`) onto the import path; also set `"simulate_data": False`.

Build the file as follows. `data_LMK` enumerates *groups* of coefficients — each row `i` names
one group, and the three `…_dataLMKindex-{i}` datasets carry that group's contents. Most people
want one LMK per group, i.e. `n_i = 1` for every `i`.

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `data_LMK` | `(N_L, 3)` | int | one `[L, M, K]` row per group; `N_L` sets the loop bound |
| `fit_LMK_dataLMKindex-{i}` | `(n_i, 3)` | int | the `[L, M, K]` triplets inside group `i` |
| `fit_coeffs_dataLMKindex-{i}` | `(n_q, n_i)` | float | C_lmk(q). **Stored q-major** and transposed on read |
| `fit_coeffs_cov_dataLMKindex-{i}` | `(n_q, n_i, n_i)` | float | covariance per q. **Only the diagonal is used** |
| `fit_axis` | `(n_q,)` | float | the q axis (Å⁻¹), multiplied by `q_scale` on read |

`{i}` runs `0 … N_L-1`. Note the axis conventions: the coefficient array is `(n_q, n_i)` — q
first — while everything downstream is `(n_lmk, n_q)`; `load_data_h5_file` does the transpose
for you. The off-diagonal covariance is read and then discarded, because the likelihood
(`gaussian_log_likelihood`, `:1658`) is diagonal.

A minimal writer for six independent coefficients:

```python
import h5py, numpy as np

lmk   = np.array([[2,0,0],[2,0,2],[4,0,0],[4,0,2],[4,0,4],[6,0,0]], dtype=int)
q     = np.linspace(0.5, 10.0, 200)          # inverse Angstrom
coeff = my_C_lmk                              # shape (6, 200)
sigma = my_C_lmk_stderr                       # shape (6, 200), standard error of the mean

with h5py.File("my_data.h5", "w") as h5:
    h5["data_LMK"] = lmk
    h5["fit_axis"] = q
    for i in range(len(lmk)):
        h5[f"fit_LMK_dataLMKindex-{i}"]        = lmk[i][None, :]        # (1, 3)
        h5[f"fit_coeffs_dataLMKindex-{i}"]     = coeff[i][:, None]      # (n_q, 1)
        h5[f"fit_coeffs_cov_dataLMKindex-{i}"] = (sigma[i]**2)[:, None, None]  # (n_q, 1, 1)
```

Then, in `parameters.py`:

- `"data_fileName"`: the path above.
- `"simulate_data": False`.
- `"isMS"`: `True` if your coefficients are **already divided by the atomic scattering**,
  otherwise `False` and `prune_data` will divide for you.
- `"dom": None` to adopt the file's `fit_axis`.
- `"fit_range"`: the q window the likelihood is computed over.
- `"fit_bases"`: which `[L, M, K]` to actually fit. **Each triplet must match exactly one row**
  of `data_LMK` — `prune_data` calls `sys.exit(0)` (no traceback) if a triplet matches zero or
  more than one.
- `"I_scale"`: supply this. With real data and no `I_scale`, `calculate_I0:2144` calls
  `self.fig_I0`, a method that does not exist (typo for `fit_I0`).

Your ADMs must cover the same `eval_times` as the measurement, and `modules/NO2.py:get_ADMs`
hardcodes `"NO2"` in its path, so a different molecule needs its own `get_ADMs` in its own
module (see `modules/module_template.py`).

---

## 11. Gotchas

- **Run the scripts from inside `NO2/`.** `modules/c_calc_extensions.py:16` loads the shared
  library from the **cwd-relative** `./cpp_extensions/lib/`, at import time — so it fires even
  for `calc_type=1`.
- **Never pass `--multiProc_ind`** unless you have re-checked both sweep lists. It selects an
  entry from a hardcoded `options` list, and the two scripts' lists must agree or
  `mode_search.py` will look for a backend that does not exist. Also note the `dom` placed in
  `options` is immediately discarded, because the following `setup_dom()` recomputes `dom` from
  `fit_range` and `q_per_pix`.
- **`build_posterior.py` must finish before `mode_search.py`.** The latter calls
  `setup_sampler(..., expect_file=True)` and raises `RuntimeError` if the backend is missing.
- **The simulated-data cache is keyed only on `get_fileName`.** `fit_bases`, `sim_thetas`,
  `eval_times`, `probe_FWHM` and `ENSEMBLE_GRID_N` are **not** in the name, so changing any of
  them silently reuses stale coefficients. Delete `output/saved_simulations/` after any such
  change. (`temperature`, `intensity`, `fit_range` and the error model *are* in the name.)
- **A too-short chain yields an empty result.** `get_mcmc_results` discards `3·max(τ)` and thins
  by `max(τ)`, so if `iteration < 4·max(τ)` you get an empty array and `mode_search.py:95` fails
  in `np.unique`. Budget at least ~10·τ; τ here settles near 40.
- **`emcee` will warn** that the chain is shorter than 50·τ on these fast settings. That is
  expected; the run is a demonstration, not a converged physics result. `has_converged` is
  recorded in the backend — check it before quoting numbers.
- **The `51` appended to `sim_thetas`** (`parameters.py:146-147`) is intended as the ensemble
  grid size but never takes effect: every consumer strips it with `[:-1]`. The grid size is
  `ENSEMBLE_GRID_N` in `modules/NO2.py`. Its only reader, `remove_global_offset:2209`, calls the
  generator with an `N=` kwarg the generator does not accept.
- **`calc_type` interacts with `multiprocessing`.** `setup_calculations:2485` takes the C++
  branch when `calc_type == 0` **or** `do_multiprocessing`, so `calc_type=1` is silently ignored
  whenever `multiprocessing > 1`.

---

## 12. Known remaining issues

**Everything outstanding is written up in [`issues/`](issues/)**, one file per issue, each with the
symptom, the exact location, the mechanism, a way to reproduce it and a suggested fix. Start with
[`issues/README.md`](issues/README.md) for the index and a suggested order of work.

The ones most likely to affect you:

| Issue | Why it matters |
|---|---|
| [006](issues/006-measured-data-path-broken.md) | **The measured-data path crashes** (`self.fig_I0` does not exist). One-character fix, but it is on the path that matters most. |
| [002](issues/002-L4-coefficients-anomalously-small.md) | The **L=4 coefficients come out ~10⁻⁶** while L=2 and L=6 are ~10⁻². Unresolved; needs a physics decision. If it is a bug, L=4 contributed nothing to any published fit. |
| [001](issues/001-ston-signal-to-noise-unusable.md) | `StoN` runs but cannot reach usable S/N with these ADMs — quantified, with what would fix it. |
| [003](issues/003-2dof-symmetric-path-unwired.md) | `experiment="2dof"` (symmetric NO₂) raises; `theta_to_cartesian_2dof` is dead code. |
| [004](issues/004-calc-type-1-and-2-broken.md) | `calc_type` 1 and 2 both crash, and `calc_type` is silently ignored when `multiprocessing > 1`. |
| [007](issues/007-simulated-data-cache-key-incomplete.md) | The simulated-data cache can **silently** reuse stale coefficients. |
| [011](issues/011-notebook-remaining-problems.md) | Two notebook cells shadow `plot_functions` with swapped positional arguments — a wrong figure with no error. |
| [012](issues/012-setup-sh-broken.md) | `setup.sh` has a fatal syntax error and cannot run at all. |
