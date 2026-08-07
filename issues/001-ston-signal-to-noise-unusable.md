# 001 — `StoN` error model cannot reach usable signal-to-noise with the in-repo ADMs

**Severity** P1 (blocks the paper's own noise model)
**Area** error model / ADMs
**Status** open — mechanism understood and quantified, no fix

## Symptom

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
well-conditioned posterior. See `how_to_run.md` §8.

## Related

- [002](002-L4-coefficients-anomalously-small.md) — the L=4 rows above
- [007](007-simulated-data-cache-key-incomplete.md) — why `save_sim_data` must be bypassed to
  reproduce
