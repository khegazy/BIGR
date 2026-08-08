# 017 — `weight_avg_search` returns a Θ\* with lower likelihood than a typical posterior sample

**Severity** P2 (the reported Θ\* can be worse than doing nothing)
**Area** mode search
**Status** open

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
