# 017 — `weight_avg_search` returns a Θ\* with lower likelihood than a typical posterior sample

**Severity** ~~P2~~ → **P3 (latent)**
**Area** mode search
**Status** ❌ **NOT REPRODUCIBLE — the observed failure was caused by
[002](002-L4-coefficients-anomalously-small.md), not by the mode search.**

Re-measured 2026-08-08 on a valid posterior (3000 × 32, asymmetric NO₂, PDF model,
`constant_sigma = 0.0962`, q ∈ [0.5, 10], τ ≈ 71). `weight_avg_search` converged in **18 s**
(10 iterations) and returned a Θ\* that is essentially the global maximum:

| point | logL | max abs deviation from truth |
|---|---|---|
| ground truth Θ | 0.0 (identity — the simulated data is noiseless) | — |
| **mode search Θ\*** | **−3.8 × 10⁻⁸** | **3.7 × 10⁻⁶** |
| posterior median | −1.9 × 10⁻¹ | 1.3 × 10⁻² |

```
Theta*  = [1.350000, 0.030001, 1.050000, 0.020000, 2.340000, 0.009996]
truth   = [1.35,     0.03,     1.05,     0.02,     2.34,     0.01    ]
```

Θ\* now beats the posterior median by a factor ~5 × 10⁶ in likelihood, which is exactly the
behaviour the routine is supposed to have and the opposite of what was recorded below. So the
centroid estimator was never the defect — it was being fed a corrupt landscape.

**What survives.** The *structural* criticism is still valid as a latent risk, and is why this file
is kept rather than deleted: a probability-weighted centroid coincides with the mode only for a
symmetric, unimodal posterior. Here it works because the corrected posterior is close to Gaussian
near its peak. On a genuinely skewed or multimodal posterior — measured data, low SNR, or the
prior-dominated regime of [018](018-posterior-prior-dominated-at-low-information.md) — it can still
fail, and it would still fail *silently* without fix 1 below. Fix 1 (the result guard) is cheap and
worth doing regardless; fixes 2–3 are unnecessary at present.

The reporting half of fix 1 is already in place: `weight_avg_search` now prints Θ\* alongside the
best sampled log-probability (added with [008](008-mode-search-no-iteration-cap.md)), so a
worse-than-samples Θ\* is at least visible. The automatic fallback is not.

## Original report (superseded — the numbers below are from a corrupt forward model)

## Symptom

On a converged 2-hour run (asymmetric NO₂, PDF model, `constant_sigma = 0.163`, q ≤ 5 Å⁻¹,
2400 × 32 samples), `mode_search.py` converged — the last two iterations of `ths_mean_history` are
bit-identical — and returned a Θ\* that fits the data **worse than the posterior median**, and far
worse than the ground truth:

| point | logL |
|---|---|
| ground truth Θ | **0.000** (the global maximum) |
| posterior median | −0.902 |
| **mode search Θ\*** | **−8.734** |

Θ\* is supposed to be the *mode* — the most probable Θ. Returning a point ~9 in logL below truth,
and ~8 below an ordinary posterior sample, means the search is not finding a maximum at all.

```
Theta*  = [1.4756, 0.0756, 1.2698, 0.0666, 1.8266, 0.1700]
truth   = [1.3500, 0.0300, 1.0500, 0.0200, 2.3400, 0.0100]
```

## Cause

`weight_avg_search` (`modules/mode_search.py:176`) does not maximise anything. It computes a
**probability-weighted average** of sampled Θ, iterated to a fixed point:

```python
scale = np.exp(log_prob_dist_raw[sort_inds[:N_samples]]) + 1e-10
ths_mean = np.sum(ths_dist_raw[sort_inds[:N_samples]].transpose()*scale, 1)/np.sum(scale)
```

A weighted centroid equals the mode only for a symmetric, unimodal distribution. This posterior is
strongly skewed and degenerate (the widths run up against their prior bound, see below), so the
centroid lands in a region of high *volume* rather than high *density* — exactly the failure mode the
paper warns about for 1-D marginals, but here it affects the estimator that was meant to avoid it.

Two aggravating details:

- The search is seeded only from `N_mode_samples` (25 here) of the highest-log-prob samples of a
  chain that never visited the neighbourhood of the true maximum, so it cannot recover it.
- There is no check that the returned Θ\* actually improves on its starting point. The routine can —
  and here does — move *downhill* and then declare convergence.

## Suggested fix

1. **Guard the result.** Cheapest and most valuable: after convergence, compare
   `log_likelihood(Θ*)` against the best sampled Θ and against the seed points, and either warn
   loudly or return the best known point instead:
   ```python
   ll_star = extractor.log_likelihood(ths_mean[None, :])[0]
   ll_best = np.max(log_prob_sampled)
   if ll_star < ll_best:
       print("WARNING: mode search returned logL {:.3f}, worse than the best sampled "
             "point {:.3f}; returning the sample instead".format(ll_star, ll_best))
   ```
   Without this the failure is silent, which is how it survived to now.
2. **Actually maximise.** Follow the weighted-average step with a local optimiser
   (`scipy.optimize.minimize` on `-logL`, Nelder–Mead is fine for 6 parameters) started from the
   best sampled point. That is what "find the Θ that maximises the posterior" requires.
3. **Multi-start.** Seed from several well-separated high-probability samples and keep the best
   result, since the landscape is multimodal.

## Related

- [008](008-mode-search-no-iteration-cap.md) — the same routine has no iteration cap.
- [018](018-posterior-prior-dominated-at-low-information.md) — the degenerate, prior-dominated
  posterior that makes the centroid estimator fail here.
