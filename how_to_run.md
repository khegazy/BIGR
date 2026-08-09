# How to run the BIGR NO₂ analysis

This guide takes you from a fresh copy of this repository to finished plots. It assumes you can open
a terminal and copy-paste commands, but nothing more than that.

**What the code does.** It takes gas-phase diffraction data and retrieves the molecular geometry
probability distribution |Ψ(R)|² — the bond lengths and angle, *and* how much each of them varies.
The method is described in `BIGR_paper.pdf` (Hegazy et al., *Communications Physics* **6**, 325
(2023), [doi:10.1038/s42005-023-01420-9](https://doi.org/10.1038/s42005-023-01420-9)). Read the paper
for the physics; read this file to run the software.

**Contents**
1. [Setup — do this once](#1-setup--do-this-once)
2. [Run the analysis](#2-run-the-analysis)
3. [Make the plots](#3-make-the-plots)
4. [Check that it worked](#4-check-that-it-worked)
5. [Settings you may want to change](#5-settings-you-may-want-to-change)
6. [Symmetric vs. asymmetric NO₂](#6-symmetric-vs-asymmetric-no2)
7. [Choosing how noisy the data is](#7-choosing-how-noisy-the-data-is)
8. [Using your own measured data](#8-using-your-own-measured-data)
9. [What is inside the output files](#9-what-is-inside-the-output-files)
10. [If something goes wrong](#10-if-something-goes-wrong)
11. [Known problems](#11-known-problems)

> Looking for the list of code changes that were needed to revive this repository? That is in
> [`CHANGES.md`](CHANGES.md). You do not need it to run anything.

---

## 1. Setup — do this once

You need Python 3.10 and a C++ compiler (on macOS, Apple's command-line tools are enough). Open a
terminal in the folder containing this file, then run the four steps below **in order**.

### Step 1 — create the Python environment

This repository is set up for [`uv`](https://docs.astral.sh/uv/), a fast Python installer. If you
don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

If a `.venv/` folder already exists, skip the first line.

> **Why the odd `--python .venv/bin/python`?** A `uv`-created environment deliberately has no `pip`
> inside it, so the usual `pip install` will not work. Always install with `uv pip install --python
> .venv/bin/python …`. If you prefer conda or plain venv, `requirements.txt` works there too with
> ordinary `pip install -r requirements.txt`.

### Step 2 — run the setup script

```bash
bash setup.sh
```

This creates the output folders, checks that the required data files are present, arranges the axis
distribution moments (ADMs) into the layout the code reads, and compiles the C++ extension for your
machine. It is safe to run more than once.

**How to tell it worked.** The last lines should report that the C++ extension was built, and should
name a file ending in `c_calc_extensions.so`. If instead you see `WARNING: the C++ build failed`,
see [§10](#10-if-something-goes-wrong).

> The compiled `.so` is not stored in the repository on purpose — it is machine-specific, and the
> version that used to be committed was a Linux binary that could not load on a Mac. Likewise the
> staged ADM folder is rebuilt rather than stored, because it is only a rearrangement of data already
> in the repository. This is why Step 2 is not optional.

### Step 3 — check that the physics is intact

```bash
MPLBACKEND=Agg .venv/bin/python scripts/test_physics.py
```

Expect **`14 passed, 0 failed`**. This takes about a minute and verifies the parts that are easy to
break silently: the molecular frame rotation, agreement between the three calculation backends, the
ordering of the anisotropy coefficients, the spherical harmonics, and the likelihood. **If anything
fails, stop and fix it before running an analysis** — a failure here means the numbers coming out of
a run cannot be trusted.

### Step 4 — always set `MPLBACKEND=Agg`

Every plot is written straight to a file, so no graphical window is needed. Putting
`MPLBACKEND=Agg` in front of the command avoids graphics-related crashes on headless machines and
clusters. All the commands below already include it.

---

## 2. Run the analysis

**Run these from inside the `NO2/` folder.** This matters: the compiled C++ library is loaded using a
path relative to wherever you launched Python, so it is only found from `NO2/`.

```bash
cd NO2
MPLBACKEND=Agg ../.venv/bin/python build_posterior.py     # step 1: explore the possibilities
MPLBACKEND=Agg ../.venv/bin/python mode_search.py         # step 2: find the best answer
```

**Order matters.** `build_posterior.py` explores which geometries are consistent with the data and
saves the result; `mode_search.py` reads that saved file and pins down the single best geometry. It
will stop with an error if you run it first.

**How long it takes.** With the settings as shipped, roughly **5 seconds per step** of the
exploration, and it stops after 3000 steps — so about **4–5 hours**, then under a minute for the mode
search. You can make it much faster or much slower; see [§5](#5-settings-you-may-want-to-change).

**You can safely interrupt it.** Progress is saved every 100 steps. Press Ctrl-C and re-run the same
command later, and it picks up where it left off. To start over instead, delete the `.h5` file under
`output/` first.

**Do not add `--multiProc_ind`.** That option is for submitting large sweeps on a compute cluster and
selects a different configuration from a hardcoded list — the two scripts' lists must agree or the
second will look for a file the first never wrote.

While it runs you will see lines like:

```
Sample 500: mean tau = [37.6 40.1 ...] / convergence False False
```

`tau` is how many steps it takes the exploration to "forget" where it was — smaller is better, and it
should level off rather than keep climbing. `convergence False` is normal for a long time; see
[§4](#4-check-that-it-worked).

---

## 3. Make the plots

The quickest route needs no notebook:

```bash
cd ..                                          # back to the repository root
MPLBACKEND=Agg .venv/bin/python scripts/analyse_run.py
```

This prints a summary of the run and a table of the retrieved geometry against the known truth.

For the figures, register the environment as a Jupyter kernel once, then open the notebook:

```bash
.venv/bin/python -m ipykernel install --user --name bigr --display-name "BIGR (.venv)"
.venv/bin/jupyter lab NO2/analyze_results.ipynb
```

Choose the **"BIGR (.venv)"** kernel, then run only the **first nine cells** — that is the "Fast run"
section, which draws the corner plot and the walker trajectories.

**Do not run the whole notebook.** Everything after cell 9 reproduces the paper's parameter sweeps
and expects a separate completed run for every configuration; it will take days or fail outright. Two
of those later cells also redefine plotting functions with different argument orders, which produces a
wrong figure without any error message
([issues/011](issues/011-notebook-remaining-problems.md)).

To produce the same figures without opening Jupyter:

```bash
.venv/bin/python - <<'EOF'
import json, copy
nb = json.load(open("NO2/analyze_results.ipynb"))
sub = copy.deepcopy(nb); sub["cells"] = nb["cells"][:9]
json.dump(sub, open("NO2/_fast.ipynb", "w"), indent=1)
EOF
cd NO2 && MPLBACKEND=Agg ../.venv/bin/python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=bigr --ExecutePreprocessor.timeout=1800 _fast.ipynb
rm _fast.ipynb            # the plots are already saved
```

The temporary notebook has to sit in `NO2/` because the kernel runs in the notebook's own folder.

### Where everything is saved

| Path | Contents |
|---|---|
| `output/<molecule>/<experiment>/sim/<model>/<T>K_<I>TW_<FWHM>fs/results_*.h5` | the explored possibilities (the posterior) |
| `…/mode_search_results_*.h5` | the single best geometry, Θ\* |
| `output/saved_simulations/<same path>/results_*.h5` | cached simulated data |
| `NO2/plots/<molecule>/<experiment>/…/fast_run_corner.png` | the main figure |
| `NO2/plots/<molecule>/<experiment>/…/fast_run_chains.png` | walker trajectories |
| `NO2/plots/check_jl*.png` | setup diagnostics (see [§10](#10-if-something-goes-wrong)) |

---

## 4. Check that it worked

The shipped configuration analyses **simulated** data, so the right answer is known in advance. That
makes it a genuine test: the retrieved values should match the values the data was generated from.
`scripts/analyse_run.py` prints exactly this comparison. A good result looks like:

| Θ | truth | retrieved | σ^Θ | best Θ\* | (retrieved−truth)/σ^Θ |
|---|---|---|---|---|---|
| ⟨NO⁽¹⁾⟩ | 1.35000 | 1.35019 | 0.00062 | 1.350000 | +0.32 |
| σ(NO⁽¹⁾) | 0.03000 | 0.02891 | 0.00306 | 0.030001 | −0.36 |
| ⟨NO⁽²⁾⟩ | 1.05000 | 1.04993 | 0.00067 | 1.050000 | −0.10 |
| σ(NO⁽²⁾) | 0.02000 | 0.01475 | 0.00728 | 0.020000 | −0.72 |
| ⟨∠ONO⟩ | 2.34000 | 2.33984 | 0.00088 | 2.340000 | −0.18 |
| σ(∠ONO) | 0.01000 | 0.02269 | 0.01427 | 0.009996 | +0.89 |

Distances in Å, angles in radians. **The last column is what to read**: it is how far the answer is
from the truth measured in its own uncertainty. Anything within about ±1 is a success. Here all six
are, and the mode search recovers the truth to 5–6 decimal places.

Three things to understand about this table before quoting it:

1. **The average bond lengths and angle are pinned down far better than their widths.** The
   uncertainty on ⟨NO⟩ is 0.6 mÅ, close to the 0.5 mÅ in the paper's Table 1; the uncertainty on the
   widths σ(·) is five to a hundred times larger in relative terms, and their distributions are
   lopsided rather than bell-shaped. That is expected — the paper (Figs. 6a, 8) shows the widths are
   the hardest quantities to measure and the first to degrade as data quality falls. Do not present
   the width numbers as a precision result.
2. **`has_converged` will say `False`, and that is not necessarily a failure.** The code demands that
   the run be 100× longer than `tau` before it says `True`, which needs about 6800 steps. At the
   shipped 3000 steps the exploration has settled down (`tau` levels off near 71 and the run is 39×
   longer than it) but has not met that strict bar. Either raise `max_iterations` to ~7000 and wait
   ~10 hours, or describe the result as *equilibrated* rather than *converged*.
3. **A short run that looks perfect proves nothing.** The walkers start *at* the true answer, so for
   the first several hundred steps they have simply not moved away yet. Early runs looked flawless
   while the underlying calculation was in fact badly wrong. Always check `tau` and the acceptance
   fraction, both printed by `scripts/analyse_run.py`, before believing a result.

What `scripts/analyse_run.py` reports, and what to want:

| Quantity | Healthy value | Meaning |
|---|---|---|
| acceptance fraction | 0.2 – 0.5 | fraction of proposed steps accepted |
| `tau` / steps | falling, ideally < 0.01 | if it stays flat, the run is drifting rather than sampling |
| (retrieved − truth)/σ^Θ | within ±1 | correctness, for simulated data |
| widths near 0.5 | **bad sign** | the answer is being set by the prior, not your data ([issues/018](issues/018-posterior-prior-dominated-at-low-information.md)) |

---

## 5. Settings you may want to change

Everything lives in **`NO2/parameters.py`**. Open it in any text editor and change the values in the
dictionary near the top.

> **One trap worth knowing.** `get_parameters()` sets `multiprocessing`, `Nwalkers` and `run_limit`
> near the top of the dictionary and then **overwrites all three** further down, in the
> `if density_model == …` block (`parameters.py:95-102`). Editing the first copy does nothing. Edit
> the one inside the block — for the default `"PDF"` model that is the `else` branch.

The settings most people want:

| Setting | Shipped value | What it does |
|---|---|---|
| `max_iterations` | 3000 | How many steps to run. **The main speed control.** 500 finishes in ~45 min and is fine for checking the pipeline works; 7000 gives a formally converged run. |
| `fit_range` | `[0.5, 10]` | The q range (Å⁻¹) used in the fit. Wider is better but slower. Do not go below 0.5 — the C++ maths is unstable there. |
| `q_per_pix` | `2*3.5/83` | Spacing of q points. Doubling it halves the number of points and roughly halves the runtime. |
| `Nwalkers` | 32 | How many walkers explore in parallel. Must be at least 12. Cost rises in proportion. |
| `simulate_error` | `("constant_sigma", 0.0962)` | How noisy the simulated data is — see [§7](#7-choosing-how-noisy-the-data-is). |
| `density_model` | `"PDF"` | `"PDF"` retrieves averages **and** widths (6 numbers). `"delta"` retrieves averages only (3 numbers) and is ~100× faster, but has known systematic errors in the widths — good for a first look. |
| `experiment` | `"3dof"` | Number of geometric degrees of freedom — see [§6](#6-symmetric-vs-asymmetric-no2). |
| `ADM_params["temperature"]` | 1 | Gas temperature in K. **Only 1, 10, 20, 30 and 300 exist.** Colder means stronger alignment and a much better measurement. |
| `ADM_params["eval_times"]` | `linspace(16.0, 20.0, 25)` | Which time points to use, in ps. **Must stay inside −0.2 … 40.3 ps** or the code raises an error. The 16–20 ps window is chosen because the molecular alignment peaks near 18 ps. |
| `ENSEMBLE_GRID_N` (in `modules/NO2.py`) | 19 | Fineness of the geometry grid. **The single biggest cost** — it scales as N³. 19 is accurate; 11 is ~5× faster and agrees to 0.001%. |

To reproduce the paper exactly, use `fit_range = [0.5, 10]`, `q_per_pix = 3.5/83`,
`simulate_error = ("StoN", (100, [0.5, 4]))`, `Nwalkers = 100`, and expect hours to days per
configuration.

**Two settings that look tempting but should be left alone.** `ENSEMBLE_GRID_SPAN` (7) is the width
of the geometry grid in standard deviations; reducing it to 3 makes the calculation 200× less
accurate at the same cost, because the quantity being integrated oscillates. And `multiprocessing`
must be 0 or 1 on macOS and Windows — the parallel path cannot be started there
([issues/005](issues/005-multiprocessing-broken-on-spawn.md)).

**If you change what the simulated data is** — `sim_thetas`, `fit_bases`, `eval_times`,
`ENSEMBLE_GRID_N` — the cached simulated data is now automatically detected as out of date and
regenerated. If you ever suspect a stale cache anyway, delete `output/saved_simulations/`.

---

## 6. Symmetric vs. asymmetric NO₂

**This is the easiest thing to get silently wrong.** The number of degrees of freedom must match the
molecule you mean to study, and this repository contains two different NO₂ geometries.

| | Symmetric NO₂ | Asymmetric ("symmetry-broken") NO₂ |
|---|---|---|
| Geometry file | `NO2/XYZ/NO2.xyz` | `NO2/XYZ/NO2_symbreak.xyz` |
| Bond lengths | both N–O = **1.1934 Å** | N–O = **1.35 Å** and **1.05 Å** |
| ∠ONO | 2.337 rad (133.9°) | 2.337 rad |
| Unknowns | one shared bond length + angle | each bond separately + angle |
| `molecule` | `"NO2"` | `"NO2_symbreak"` |
| `experiment` | `"2dof"` | `"3dof"` |
| Works today? | **No** — see below | **Yes** |

The rule: **if both bonds are the same length, there is one bond unknown; if they differ, each bond
gets its own.** `get_parameters()` already links these correctly — `molecule` picks the geometry file
and `experiment` picks the list of unknowns — so if you change one, change the other.

**The symmetric (`"2dof"`) path does not currently work.** The function that builds a symmetric
geometry exists but nothing calls it, so `experiment="2dof"` combines symmetric assumptions with
asymmetric geometry generation and fails with an `IndexError`.
[issues/003](issues/003-2dof-symmetric-path-unwired.md) records exactly what to add.

The shipped configuration is the **asymmetric** case, which is also the paper's headline simulated
result (Table 1: ⟨NO⁽¹⁾⟩ = 1.3500 Å, ⟨NO⁽²⁾⟩ = 1.0500 Å, ∠ONO = 2.34 rad).

---

## 7. Choosing how noisy the data is

When analysing simulated data you choose how realistic the noise should be, via `simulate_error`.

| Setting | Error bars | Use it when |
|---|---|---|
| `("StoN", (SNR, q_range))` | Realistic: different for every coefficient **and** every q, computed by adding photon-counting noise to simulated diffraction images | You want publication-grade results. This is the paper's model. |
| `("constant_background", σ)` | Grows with q, the physically right shape for detector background | You want realistic behaviour without the extra machinery. Note the number is 1/(signal-to-noise), not the signal-to-noise. |
| `("constant_sigma", σ)` | One value everywhere | Getting started. **This is what ships.** |
| `("data", (order, Wn))` | Taken from a real measurement | Needs your own measured data file; see [§8](#8-using-your-own-measured-data). |

**Picking σ for `constant_sigma`.** σ is the assumed error on every coefficient, so it sets the
resolution directly. To match the paper's signal-to-noise convention:

| paper SNR | use σ = |
|---|---|
| 25 | 0.385 |
| **100** (the paper's standard) | **0.0962** ← shipped |
| 400 | 0.0241 |

σ = 0.0962 gives 0.63 mÅ on ⟨NO⁽¹⁾⟩ and 1.1 mrad on ⟨∠ONO⟩, against the paper's 0.5 mÅ and 1 mrad.

**One coupled setting.** `init_thetas_std_scale` controls how spread out the walkers start. It must
roughly match how sharp the answer is. Too wide and almost every step is rejected and the walkers
freeze (`tau` comes back as `nan`); too narrow and the run wastes thousands of steps expanding. At
σ = 0.0962 the shipped value of `5e-4` is right. If you change σ a lot, expect to change this too.

---

## 8. Using your own measured data

You need one HDF5 file. Point `data_fileName` in `parameters.py` at its **full path** and set
`simulate_data` to `False`.

The file is flat — every dataset sits at the top level, no groups. `data_LMK` lists *groups* of
coefficients; most people want one coefficient per group, so `n_i = 1` throughout.

| Dataset | Shape | Type | Meaning |
|---|---|---|---|
| `data_LMK` | `(N_L, 3)` | int | one `[L, M, K]` row per group |
| `fit_LMK_dataLMKindex-{i}` | `(n_i, 3)` | int | the `[L, M, K]` values in group `i` |
| `fit_coeffs_dataLMKindex-{i}` | `(n_q, n_i)` | float | your C coefficients. **Note q comes first** |
| `fit_coeffs_cov_dataLMKindex-{i}` | `(n_q, n_i, n_i)` | float | the covariance at each q. **Only the diagonal is used** |
| `fit_axis` | `(n_q,)` | float | the q axis in Å⁻¹ |

`{i}` runs from `0` to `N_L - 1`. Here is a complete script for six independent coefficients — adapt
the three variables at the top:

```python
import h5py, numpy as np

lmk   = np.array([[2,0,0],[2,0,2],[4,0,0],[4,0,2],[4,0,4],[6,0,0]], dtype=int)
q     = np.linspace(0.5, 10.0, 200)   # your q axis, inverse Angstrom
coeff = my_C_lmk                      # your coefficients, shape (6, 200)
sigma = my_C_lmk_stderr               # their standard errors, shape (6, 200)

with h5py.File("my_data.h5", "w") as h5:
    h5["data_LMK"] = lmk
    h5["fit_axis"] = q
    for i in range(len(lmk)):
        h5[f"fit_LMK_dataLMKindex-{i}"]        = lmk[i][None, :]                # (1, 3)
        h5[f"fit_coeffs_dataLMKindex-{i}"]     = coeff[i][:, None]              # (n_q, 1)
        h5[f"fit_coeffs_cov_dataLMKindex-{i}"] = (sigma[i]**2)[:, None, None]   # (n_q, 1, 1)
```

Then set these in `parameters.py`:

- `"data_fileName"` — the full path to the file you just wrote.
- `"simulate_data": False`.
- `"isMS"` — `True` if your coefficients are **already divided by the atomic scattering**, `False`
  if not (the code will divide for you).
- `"dom": None` — take the q axis from your file.
- `"fit_range"` — the q window to actually fit over.
- `"fit_bases"` — which `[L, M, K]` to fit. **Each one must match exactly one row of `data_LMK`.**
  If a value matches zero rows or more than one, the program exits immediately with no error
  message — an easy problem to mistake for a crash.
- `"I_scale"` — the overall scale between your data and the calculation. If you leave it out the
  code fits it for you.

Your ADMs must cover the same times as your measurement. For a molecule other than NO₂ you also need
your own `get_ADMs`, `get_scattering_amplitudes` and geometry functions — copy
`modules/module_template.py` and follow `modules/NO2.py` as the worked example.

**Two known issues on this path**, both worth reading before you trust the output:
[issues/019](issues/019-prefactor-convention-differs-from-eq21.md) (a normalisation convention that
differs from the paper's Eq. 21 and matters for imported data) and
[issues/013](issues/013-get-adms-molecule-hardcoded.md).

---

## 9. What is inside the output files

All output files are flat HDF5 — datasets at the top level, no groups. You only need this section if
you want to read the results yourself instead of using the notebook.

**The main results file, `results_*.h5`** — rewritten from scratch at every checkpoint:

| Dataset | Shape | Notes |
|---|---|---|
| `chain` | `(n_steps, nwalkers, ndim)` | every geometry that was tried |
| `log_prob` | `(n_steps, nwalkers)` | how well each one fit |
| `nwalkers`, `ndim` | scalars | `ndim` is 6 for PDF/3dof, 3 for delta/3dof |
| `accepted` | `(nwalkers,)` | accepted-step counts |
| `has_converged` | scalar | see [§4](#4-check-that-it-worked) |
| `tau_convergence` | `(n_batches, ndim)` | `tau` history, one row per checkpoint |
| `autocorr_times` | `(ndim,)` | most recent `tau` |
| `filtered_chain` | `(n_thin, nwalkers, ndim)` | thinned chain, or `[False]` if the run is too short |

**The mode-search file, `mode_search_results_*.h5`:**

| Dataset | Shape | Notes |
|---|---|---|
| `ths_mean` | `(ndim,)` | **Θ\*, the best geometry** — usually the only one you need |
| `ths_var`, `ths_std` | `(ndim,)` | spread of the weighted sample |
| `ths_mean_history`, `ths_std_history` | `(n_iters, ndim)` | per-iteration history |
| `ths_sampled` | `(ndim, ≤10000)` | geometries evaluated |
| `log_prob_sampled`, `chiSq_sampled` | `(≤10000,)` | their scores |

It sits in the same folder as the results file, with `mode_search_` added to the front of the name.

**The cache, `output/saved_simulations/…/results_*.h5`** holds `input_data_coeffs`,
`input_data_coeffs_var` and sometimes `experimental_var`, each `(n_lmk, n_dom)`. It also stores a
`sim_key` fingerprint so that changing any setting the data depends on is detected and the cache is
regenerated.

Reading the best answer in three lines:

```python
import h5py
with h5py.File("output/.../mode_search_results_....h5") as f:
    theta_star = f["ths_mean"][:]     # [<NO1>, sig(NO1), <NO2>, sig(NO2), <ONO>, sig(ONO)]
```

---

## 10. If something goes wrong

**`ModuleNotFoundError` / `ImportError`** — the environment is missing or not being used. Every
command must use `.venv/bin/python` (or `../.venv/bin/python` from inside `NO2/`), not a bare
`python`. Re-run Step 1.

**`OSError: … c_calc_extensions.so … image not found`** — either the C++ extension was not built
(re-run `bash setup.sh`), or you are running from the wrong folder. `build_posterior.py` and
`mode_search.py` must be launched **from inside `NO2/`**, because that library is found by a path
relative to your current folder.

**The C++ build fails** — you need compiler command-line tools (`xcode-select --install` on macOS).
As a fallback, set `calc_type: 1` in `parameters.py` to use the pure-Python maths: correct
everywhere, but much slower.

**`mode_search.py` fails immediately** — `build_posterior.py` has not produced a results file yet,
or has not run long enough. It needs to be at least ~4× `tau`; if it is shorter the thinned chain is
empty and the mode search fails while reading it. Let the first script run longer.

**`emcee` warns that the chain is shorter than 50×tau** — expected, and not an error. See point 2 in
[§4](#4-check-that-it-worked).

**Results look wrong, or `tau` climbs forever** — check `NO2/plots/check_jl*.png`. These show the
spherical Bessel functions the calculation depends on; the curves should be smooth and the residual
negligible across your `fit_range`. If they blow up at small q, raise the lower end of `fit_range`
(0.5 Å⁻¹ is a safe floor). Then re-run `scripts/test_physics.py`.

**Everything is very slow** — reduce `max_iterations` first, then `ENSEMBLE_GRID_N` from 19 to 11
(~5× faster, 0.001% less accurate), then double `q_per_pix`. Setting `density_model: "delta"` is
~100× faster if you only need the average geometry.

**You changed a setting and the results did not change** — you probably edited the copy of
`multiprocessing`/`Nwalkers`/`run_limit` that is overwritten later in the file. See the note at the
top of [§5](#5-settings-you-may-want-to-change).

---

## 11. Known problems

Every known defect is written up in **[`issues/`](issues/)** — one file each, with the symptom, where
it is, why it happens, how to reproduce it, and a suggested fix.
[`issues/README.md`](issues/README.md) is the index.

The ones most likely to affect you:

| Issue | Why it matters |
|---|---|
| [019](issues/019-prefactor-convention-differs-from-eq21.md) | A coefficient normalisation differs from the paper's Eq. 21 by an L-dependent factor. **Matters if you import your own measured data**, and needs a decision from the authors. |
| [003](issues/003-2dof-symmetric-path-unwired.md) | Symmetric NO₂ (`experiment="2dof"`) does not run at all. |
| [005](issues/005-multiprocessing-broken-on-spawn.md) | `multiprocessing` must be 0 or 1 on macOS and Windows. |
| [011](issues/011-notebook-remaining-problems.md) | Later notebook cells can draw a **wrong figure with no error** — another reason to stop at cell 9. |
| [018](issues/018-posterior-prior-dominated-at-low-information.md) | With low-quality data the widths can be set by the prior rather than by your measurement. How to spot it. |
| [004](issues/004-calc-type-1-and-2-broken.md) | `calc_type: 2` does not work; `calc_type` is ignored when `multiprocessing > 1`. |
