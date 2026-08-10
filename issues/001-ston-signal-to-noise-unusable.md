# 001 — [LARGELY RESOLVED] `StoN` signal-to-noise with the in-repo ADMs

**Severity** ~~P1~~ P3
**Area** error model / ADMs
**Status** **largely resolved 2026-08-08** — the original diagnosis was dominated by the
centre-of-mass bug.

> ## Re-measured after fixing the centre-of-mass bug
>
> The S/N figures below were taken while `rotate_to_principalI` scaled every molecule by
> 1/total_mass ([002](002-L4-coefficients-anomalously-small.md)). The C coefficients were therefore
> ~500× too small and, because the tiny distances destabilised the C++ Bessel recursion, partly
> noise. The error propagation was fine; the signal it was being compared against was not.
>
> Re-measured with the bug fixed, `StoN` at the paper's standard SNR = 100 with only 25 eval_times:
>
> | LMK | S/N (fixed) | S/N (with the bug) |
> |---|---|---|
> | [2 0 0] | **144.9** | 0.055 |
> | [4 0 0] | **28.7** | ~3e-6 |
> | [6 0 0] | **16.1** | 0.009 |
> | [2 0 2] | 2.2 | 0.015 |
> | [4 0 2] | 2.2 | ~2e-7 |
> | [4 0 4] | 0.006 | ~4e-7 |
>
> **So `StoN` is usable, and the ADMs in this repository are adequate.** The k = 0 coefficients are
> well measured. The much lower S/N on k ≠ 0 is expected physics rather than a defect: those terms
> are constrained by the azimuthal ADMs (A^L_{0K} with K ≠ 0), which are genuinely smaller than the
> K = 0 moments — visible directly in the ADM files.
>
> **What survives:** the mechanism section below is a correct description of how
> `simulate_error_StoN` propagates error (`Var(C) = (AᵀWA)⁻¹` with mean-subtracted ADMs), and the
> observation that variance falls only as 1/N_times so adding time points is an inefficient lever.
> Both remain useful. **What does not survive:** the conclusion that usable S/N would need ~10⁶ time
> points, and the recommendation to convert the older MATLAB ADM set — neither is necessary.
>
> `StoN` is now the right choice for a paper-faithful run, and is preferable to `constant_sigma`
> because its error bars are per-coefficient and per-q rather than uniform.

---

## Original symptom (with the centre-of-mass bug present)

With `simulate_error = ("StoN", (100, [0.5, 4]))` the pipeline now runs end to end — ADM import,
2-D diffraction simulation, Legendre refit, error propagation, MCMC — but the resulting
molecular-frame error bars are **larger than the coefficients themselves**, so the likelihood is
almost flat and the posterior is prior-dominated. The retrieval is meaningless.

Measured mean |C_lmk| / σ_lmk inside `fit_range`, asymmetric NO₂, ADMs at 1 K over the 16–20 ps
revival:

| L M K | 25 eval_times | 200 eval_times |
|---|---|---|
| 2 0 0 | 0.055 | 0.153 |
| 2 0 2 | 0.015 | 0.042 |
| 4 0 0 | 3.3e-06 | 3.0e-06 |
| 6 0 0 | 0.0093 | 0.026 |

(The L=4 rows are dominated by a separate problem — see [002](002-L4-coefficients-anomalously-small.md).)

For comparison, the paper's results use SNR 25–400 on C₀₀₀ and obtain sub-mÅ resolution.

## Mechanism

`simulate_error_StoN` (`modules/density_extraction.py:2787`) converts lab-frame variance into
molecular-frame C_lmk variance at `:2946-2953`:

```python
fit_ADMs = self.ADMs[linds,:]
fit_ADMs -= np.mean(fit_ADMs, -1, keepdims=True)
fit_var_ = np.linalg.inv(
    np.einsum('ai,ib,ci->bac', fit_ADMs, 1/s2n_var[:,:,il], fit_ADMs))
```

This is the covariance of a weighted least-squares fit of the measured anisotropy onto the ADMs:
`Var(C) = (Aᵀ W A)⁻¹` with `A` the **mean-subtracted** ADM matrix. Two consequences:

1. `Var(C) ∝ 1/A²`. Weak alignment makes `A` nearly singular and the variance explodes. The ADMs
   in this repo peak at `|A − mean| ≈ 0.07` at 1 K (and only 0.002 for L=6 at 30 K).
2. `Var(C) ∝ 1/N_times`, so **S/N only grows as √N_times**. The table above confirms this: 0.055 →
   0.153 for an 8× increase in time points is a ratio of 2.8 ≈ √8.

Extrapolating, reaching S/N ≈ 10 would need roughly **10⁶ time points**. Adding time points is
therefore not a viable fix.

## Reproduce

```bash
cd NO2
MPLBACKEND=Agg ../.venv/bin/python - <<'PY'
import numpy as np
from parameters import get_parameters
from modules.density_extraction import density_extraction
from modules.NO2 import *
for n in [25, 200]:
    p = get_parameters()
    p["simulate_error"] = ("StoN", (100, [0.5, 4]))
    p["ADM_params"]["eval_times"] = np.linspace(16.0, 20.0, n)
    p["plot_setup"] = False
    p["save_sim_data"] = None            # eval_times is not in the cache file name
    ex = density_extraction(p, get_molecule_init_geo, get_scattering_amplitudes,
            log_prior=log_prior_3dof_gauss,
            density_generator=molecule_ensemble_generator,
            ensemble_generator=molecule_ensemble_generator, get_ADMs=get_ADMs)
    sn = np.nanmean(np.abs(ex.data_coeffs)/np.sqrt(ex.data_coeffs_var), axis=-1)
    print(n, "eval_times ->", dict(zip(map(str, ex.data_LMK), np.round(sn, 5))))
PY
```

## What would actually fix it, in order of leverage

1. **Stronger alignment.** `Var(C) ∝ 1/A²`, so this is quadratic. Higher pump fluence (paper
   Fig. 6c) or a genuinely deeper revival than the one in this ADM set.
2. **Wider q and more C_lmk.** `fit_range` in the fast configuration is only `[0.5, 5]` versus the
   paper's `[0.5, 10]`–`[0.5, 20]`, and each additional C_lmk adds an angular constraint
   (paper, Discussion).
3. **A different ADM set.** The MATLAB `.dat` ADMs on the previous laptop,
   `…/density_extraction/ADMs/SLAC/10TW_100fs_{12p5K,50K}/`, span −0.2 → 800.3 ps and cover
   12.5 K and 50 K, so the original `eval_times = linspace(37.5, 41.5, 100)` window works
   unchanged. To convert: read `A{L}{K}.dat` with `np.loadtxt` (1-D, 2768 points) → `A{L}{K}.npy`,
   and `time.dat` → `times.npy`. **Use the `A` files, not `D{L}{K}.dat`** — those are
   `(225, 2768)` matrices, not the ADMs. It is worth checking whether this is the set the paper
   actually used.
4. **Regularise the ADM fit.** Supplementary Note 3 discusses regularisation for
   non-orthogonal ADMs; `(AᵀWA)⁻¹` with no regularisation is the worst case.

## Workaround in place

`NO2/parameters.py` defaults to `("constant_sigma", 0.163)`, which needs no ADMs and gives a
well-conditioned posterior. See `how_to_run.md` §7 (choosing how noisy the data is).

## Related

- [002](002-L4-coefficients-anomalously-small.md) — the L=4 rows above
- [007](007-simulated-data-cache-key-incomplete.md) — why `save_sim_data` must be bypassed to
  reproduce
