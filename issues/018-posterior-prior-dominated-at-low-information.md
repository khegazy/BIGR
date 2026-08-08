# 018 — At low information content the posterior is prior-dominated and the widths run to their bound

**Severity** P2 (not a code defect — a property of the model that will silently mislead)
**Area** priors / interpretation
**Status** open. Behaviour understood and reproducible; needs a documented guard, not a fix to the
physics.

## What happens

A 2-hour run (asymmetric NO₂, PDF model, `constant_sigma = 0.163`, q ≤ 5 Å⁻¹, 32 walkers × 2400
steps, `ENSEMBLE_GRID_N = 19`) initialised **at** the ground truth does the following:

1. Walkers leave truth within ~300–500 steps and never return.
2. The parameter distribution then becomes stationary — segment medians are flat from step 600 to
   2400 — so this is the equilibrium posterior, not a transient.
3. The retrieved widths settle 5–20× above truth, and in the walker-trajectory plot
   (`fast_run_chains.png`) σ(∠ONO) visibly **piles up against its prior ceiling** of
   `sig_max = 0.5` (`modules/NO2.py:65`).

| Θ | truth | posterior median | σ^Θ | (med−truth)/σ^Θ |
|---|---|---|---|---|
| ⟨NO⁽¹⁾⟩ | 1.35000 | 1.89457 | 0.34396 | +1.58 |
| σ(NO⁽¹⁾) | 0.03000 | 0.13493 | 0.10416 | +1.01 |
| ⟨NO⁽²⁾⟩ | 1.05000 | 1.44789 | 0.28488 | +1.40 |
| σ(NO⁽²⁾) | 0.02000 | 0.10586 | 0.07898 | +1.09 |
| ⟨∠ONO⟩ | 2.34000 | 1.87766 | 0.46421 | −1.00 |
| σ(∠ONO) | 0.01000 | 0.19357 | 0.13571 | +1.35 |

**Crucially, this is not a fitting error.** Truth remains the global likelihood maximum:
`logL(truth) = 0` exactly, and no sample among 2400 × 32 = 76 800 beat it (best −0.126). The forward
model is right; the posterior mass simply is not where the peak is.

## Why

It is a volume effect. At σ = 0.163 with q ≤ 5 Å⁻¹ the diffraction fringes carry little information,
so **broadening the ensemble costs almost nothing in likelihood** — a broad ensemble washes out
fringes that are already poorly measured. Meanwhile the prior on each width is flat over
`[0, 0.5]`, so the large-width region carries vastly more prior volume than the narrow region around
truth. Posterior mass ∝ likelihood × volume, and volume wins.

The signature to look for is exactly what the chain plot shows: **a width parameter accumulating at
its prior bound**. That means the data does not constrain it and the answer is being set by
`sig_max`, an arbitrary number in the prior, not by the measurement.

## Why it matters

The failure is silent. Every automated check passes:

- acceptance fraction 0.257 (healthy),
- the chain is stationary,
- `logL(truth) = 0` — the usual self-consistency check,
- no sample beats truth.

Yet the reported Θ is wrong by 0.5 Å in a bond length. Someone running this on measured data, where
there is no truth to compare against, would have no indication anything was amiss. And because
`weight_avg_search` also fails on this landscape ([017](017-mode-search-returns-worse-than-median.md)),
Θ\* does not rescue it either.

## Suggested guards

1. **Warn when any width parameter's posterior presses against its prior bound.** A simple check —
   e.g. more than 5 % of samples within 1 % of `sig_max` — would have flagged this run immediately.
2. **Report the prior-sensitivity of the answer.** Re-running with a different `sig_max` and seeing Θ
   move is a direct demonstration that the result is prior- rather than data-driven.
3. **Quote `logL(Θ_reported)` alongside Θ.** Here it would have shown −0.9 (median) and −8.7 (Θ\*)
   against a reachable 0.0, which is a clear signal that the reported point is not the best fit.
4. **Document the information floor.** The paper's own scans (Fig. 6a, 6b) show σ^Θ improving
   strongly with SNR and q. This run sits far below the paper's regime (SNR 25–400 and
   q ≤ 10–20 Å⁻¹), which is why it lands in the uninformative limit. Worth stating explicitly what
   minimum SNR/q makes the widths identifiable.

## Caveat on the σ used here

`constant_sigma = 0.163` is not a physically calibrated noise level — it was derived from likelihood
curvature at a coarse ensemble grid, where that curvature was dominated by quadrature noise
([016](016-ensemble-quadrature-error-dominates-likelihood.md)). A properly calibrated error model
(the paper uses `StoN`, currently unusable per
[001](001-ston-signal-to-noise-unusable.md)) would put the run in a much more informative regime.
This issue documents the *behaviour in the uninformative limit*, which is real and worth guarding
against, not a claim that the paper's published configuration suffers from it.
