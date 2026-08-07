# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BIGR retrieves the molecular-frame geometry probability distribution |Ψ(R,t)|² from ultrafast gas-phase
(electron/X-ray) diffraction data using Bayesian inference and deterministic ensemble anisotropy. It
decomposes measured diffraction into molecular-frame C_lmk(q) coefficients, models them with a
parameterized structure distribution P(R|Θ,C), and uses MCMC (`emcee`, Metropolis-Hastings) to build the
posterior P(Θ|C). `mode_search.py` then finds the mode Θ* of that posterior; P(R|Θ*,C) is the final
approximation of |Ψ(R)|².

**The method paper is in this repo: `BIGR_paper.pdf`** — Hegazy et al., "Applying Bayesian inference and
deterministic anisotropy to retrieve the molecular structure |Ψ(R)|² distribution from gas-phase
diffraction experiments," *Communications Physics* **6**, 325 (2023), doi:10.1038/s42005-023-01420-9
(published version of arXiv:2207.09600). Read it for any physics/method question before guessing from code.

## Paper ↔ code map

The analysis chain (paper Methods, Figs. 3 and 5):
measured ⟨I(q,t)⟩ → lab-frame anisotropy components B_l^m(q,t) (Eq. 17) → fit onto axis distribution
moments (ADMs, Eq. 1) → molecular-frame C_lmk(q) (Eq. 3) → MCMC inversion for P(Θ|C) (Eqs. 11, 18) →
mode search for Θ* → P(R|Θ*,C) ≈ |Ψ(R)|². This repo implements the C_lmk-and-beyond stages; ADMs are
imported precomputed via `get_ADMs`.

- C_lmk forward model (Eqs. 4, 20–21): `density_extraction.calculate_coeffs_*` (scipy / cpp / ensemble
  variants). The spherical Bessel evaluation j_l(qΔR_μν) inside it is the dominant cost.
- Gaussian likelihood P(C|Θ) (Eq. 11): `density_extraction.gaussian_log_likelihood` +
  `log_likelihood`; `log_joint_probability` adds the prior (Eq. 18) and is what `emcee` samples.
- P^(N)(R|Θ,C), multivariate normal (Eqs. 9–10; Θ = means **and** widths of pairwise distances/angles):
  code name `density_model: "PDF"`, generator `molecule_ensemble_generator` — discretizes each 1D normal
  on an N-point grid (±7σ; N from the last element of `sim_thetas`, default 19) and takes the joint grid.
- P^(δ)(R|Θ,C), delta function (Eqs. 7–8; Θ = distances/angle only): code name `density_model: "delta"`,
  generator `single_molecule_generator`. ~100× faster but suffers q-range-dependent systematic errors
  (paper Fig. 8) — use as a fast first pass, trust P^(N) for widths.
- Resolution σ^Θ = std of the 1D marginal projection of P(Θ|C). Θ* is found in the full correlated
  Θ-space (paper Fig. 5 discussion), not from 1D marginals — that is why `mode_search` exists instead of
  just taking per-parameter medians.

## Setup

```bash
bash setup.sh          # creates plots/, XYZ/, output/ dirs; symlinks modules/cpp_extensions into NO2/;
                        # downloads fitting/diffraction-simulation helper modules; builds the C++ extension
```

`setup.sh` contains hardcoded paths from the original SLAC/LCLS cluster environment (e.g.
`/cds/home/k/khegazy/...`) — treat these as examples to adapt, not values to run as-is. Likewise,
`NO2/parameters.py` has hardcoded `output_dir`, `save_sim_data`, `scat_amps_dir`, and `ADM_params["folder"]`
that must point at real local paths before a run will work.

Required: Python ≥3.9.7, numpy, scipy, matplotlib, h5py, emcee ≥3.0.2, corner. The C++ extension needs a
working `g++`; if compilation fails the analysis still runs via the pure-Python/Scipy Bessel-function
path (see `calc_type` below), just much slower and effectively restricted to the delta model.

## Build / run

```bash
cd cpp_extensions/lib && make clean && make   # rebuild the C++ extension (c_calc_extensions.so)
```

There is no test suite, linter, or CI in this repo. There is also no package entry point — everything is
run as standalone scripts from the directory containing a `parameters.py`:

```bash
cd NO2
python build_posterior.py [--multiProc_ind N]   # 1) retrieve the posterior P(Theta|C) via MCMC
python mode_search.py [--multiProc_ind N]        # 2) find the mode Theta* of that posterior
```

**Order matters**: `mode_search.py` calls `setup_sampler(..., expect_file=True)` — it loads the saved
emcee backend (`<output_dir>/<name>.h5`) written by `build_posterior.py` and seeds `weight_avg_search`
from the top `N_mode_samples` samples by log-probability. Run `build_posterior.py` to convergence first.

Both scripts checkpoint: `run_mcmc` saves the emcee backend every `run_limit` steps and resumes from it on
restart; convergence requires ≥100 autocorrelation times (or `min_acTime_steps`) with <1% change in the
autocorrelation time. `--multiProc_ind` indexes into a hardcoded `options` list at the bottom of each
script (cluster array-job sweeps over q_max / SNR / ADM params) — omit it to run the base parameters.

## Architecture

**Per-experiment folder pattern.** `NO2/` is the one implemented example experiment; a new experiment is
created the same way: a folder containing `parameters.py`, symlinks to the shared `modules/` and
`cpp_extensions/` directories, and a molecule-specific module analogous to `modules/NO2.py`. `setup.sh`
wires up these symlinks. (The README mentions `parameters_template.py`, which is currently absent —
`NO2/parameters.py` is the de facto template.)

**Three-file contract per experiment**, mirrored between the generic templates at repo root
(`build_posterior_template.py`, `mode_search_template.py`, `modules/module_template.py`) and the concrete
NO2 implementation (`NO2/build_posterior.py`, `NO2/mode_search.py`, `modules/NO2.py`):
- `parameters.py` — a single flat dict of runtime parameters built by `get_parameters()`. This dict
  becomes `density_extraction.data_params` and is threaded through nearly every function — it also
  doubles as the mechanism for passing experiment-specific values into the generator/likelihood callbacks.
- `modules/<experiment>.py` (implements `module_template.py`'s contract) — supplies the physics/geometry
  callbacks: `single_molecule_generator`/`molecule_ensemble_generator` (sample geometries from Θ),
  log priors, `initialize_walkers`, `get_molecule_init_geo`, `get_ADMs` (axis distribution moments),
  `get_scattering_amplitudes`. Note the template names priors `log_prior_gauss`/`log_prior_delta`, but
  `modules/NO2.py` provides dof-specific variants `log_prior_{2,3}dof_{gauss,delta}` selected by the
  `experiment` parameter (`"3dof"`/`"2dof"`) together with `density_model`.
- `<experiment>/build_posterior.py` / `mode_search.py` — thin drivers: pick the callbacks from
  `density_model` + `experiment`, construct a `density_extraction` instance, and call `run_mcmc` or the
  mode-search routine.

**`modules/density_extraction.py`** (`density_extraction` class, ~3000 lines) is the core engine shared by
every experiment — it owns data import/simulation, C-coefficient calculation (Scipy and C++ backends),
the log-likelihood/log-prior/log-joint-probability used by `emcee`, MCMC checkpointing
(`setup_sampler`/`run_mcmc`/`save_emcee_backend`/`load_emcee_backend`), and results/plotting. It is
constructed with the experiment's callbacks injected as arguments rather than subclassed — new experiments
plug in by writing those callbacks, not by modifying this file.

**`modules/mode_search.py`** implements the mode-finding search (`weight_avg_search`,
`single_dim_search`) used by the `mode_search.py` drivers to locate Θ*, reusing the same
`density_extraction` instance/likelihood for consistency with the posterior retrieval. It saves progress
periodically and can be killed and restarted.

**C++ acceleration (`cpp_extensions/`)** — `src/c_calc_extensions.cpp` (declared in
`include/c_calc_extensions.h`, built by `lib/Makefile` into `lib/c_calc_extensions.so`) provides a fast
recursive spherical Bessel function evaluator and a fused C-coefficient calculation.
`modules/c_calc_extensions.py` loads the `.so` via `ctypes`/`numpy.ctypeslib` and wraps it as
`spherical_j`/`calculate_c_cpp`. Selected via the `calc_type` runtime parameter: `0` = C++ (fastest, but
the recursion is numerically unstable below q ≈ 0.5 Å⁻¹ from 1/x terms — that region is typically
detector-blocked/contaminated and should be excluded via `fit_range` anyway), `1` = Scipy (slowest,
correct everywhere), `2` = optimized pure-Python (same accuracy limits as 0, slower). After changing
`calc_type` to 0 or 2, sanity-check the `check_jl{l}_calculation.png` diagnostic plots for stability in
the q-range used.

**`modules/plot_functions.py`** holds standalone corner-plot/3D/trend plotting helpers used by the NO2
analysis and notebooks (`NO2/analyze_results.ipynb`, `NO2/plots/analysis/plot_analysis_steps.ipynb`,
`toy_mcmc/toy_mcmc.ipynb`) — diagnostics, not part of the retrieval algorithm itself.

## Key parameter semantics (`parameters.py` dict)

- `density_model`: `"PDF"` (normal model, paper's P^(N)) vs. `"delta"` (point geometry, paper's P^(δ)) —
  selects which generator/prior pair the drivers wire into `density_extraction`, and switches the
  walker/multiprocessing defaults in `get_parameters()`.
- `experiment`: `"3dof"` vs. `"2dof"` — number of structural degrees of freedom; selects the matching
  `log_prior_*` and the length/content of `sim_thetas`/`init_thetas`.
- `simulate_data` + `simulate_error`: when simulating rather than importing measured C coefficients,
  selects the noise model — `("StoN", (SNR, q_range))` (Poissonian, recommended), `("constant_sigma", σ)`,
  or `("constant_background", σ)`. (Note: the keys are `simulate_*`, not `simulated_*` as the README says.)
- `sim_thetas`: Θ used to generate the simulated "truth" C coefficients. For the PDF model its last
  element is the per-dof grid size N of the discretized ensemble (appended as 51 in `NO2/parameters.py`).
- `fit_bases`: list of `[L,M,K]` triplets selecting which anisotropy/order terms are fit; unset means all
  imported C_LMK are used. More C_lmk → better angular (θ,φ) resolution (paper, Discussion).
- `fit_range` / `dom`: q-window for the likelihood and the reciprocal-space sample points. Wider q reduces
  false Θ correlations and improves widths; SNR matters more than q-range beyond ~8 Å⁻¹ (paper Figs. 6–7).
- `isMS`: True if the imported data is already divided by the atomic scattering.
- `save_sim_data`: caches simulated C coefficients to disk — delete this cache after changing the ensemble
  generator, or stale simulated data will be reused silently.
- Mode search: `N_mode_samples` (posterior samples seeding the search), `mode_std_grid`, `mode_tolerance`.
