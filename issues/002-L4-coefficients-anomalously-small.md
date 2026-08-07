# 002 — L=4 C coefficients are ~10⁻⁶ while L=2 and L=6 are ~10⁻², unexplained

**Severity** P2 (correctness risk — needs a physics decision)
**Area** C coefficient calculation
**Status** open, **unresolved**. Flagged for review; I could not determine whether this is a bug
or a genuine cancellation.

## Symptom

For simulated asymmetric NO₂ the molecular-frame C coefficients used in the likelihood are:

| L M K | rms C_lmk | max \|C_lmk\| |
|---|---|---|
| 2 0 0 | 1.98e-02 | 4.19e-02 |
| 2 0 2 | 4.01e-02 | 8.45e-02 |
| **4 0 0** | **1.07e-06** | 2.90e-06 |
| **4 0 2** | **1.25e-06** | 3.52e-06 |
| **4 0 4** | **1.62e-05** | 4.50e-05 |
| 6 0 0 | 2.20e-02 | 1.07e-01 |

Every L=4 coefficient is **3–4 orders of magnitude smaller** than both L=2 and L=6. Since L=6 is
comparable to L=2, this is not a smooth fall-off with L — it is an L=4-specific suppression.

Consequence: the L=4 coefficients contribute essentially nothing to the likelihood, so half of the
six fitted coefficients are dead weight. If the suppression is a bug, every fit that included
L=4 — including the published ones — was effectively fitting only L=2 and L=6.

## Why it looks wrong

Paper Fig. 3c plots M₂₀ₖ, M₄₀ₖ (×2) and M₆₀ₖ (×4). Those scale factors mean M₄₀ₖ and M₆₀ₖ are
roughly 2× and 4× *smaller* than M₂₀ₖ — not 10⁴ smaller. So the figure in the paper is
inconsistent with what the code produces here.

An exact-looking cancellation holding across **all q simultaneously** and for **all three K
values** (0, 2, 4) would be a remarkable coincidence. That is what makes a bug plausible.

## Why it might nevertheless be real

`C_lmk ∝ Σ_{μ≠ν} f_μ f*_ν j_l(q ΔR_μν) Y_l^{−k}(θ_μν, φ_μν)`, summed over the six ordered atom
pairs after rotation into the principal-axis frame. Y₄⁰ ∝ 35cos⁴θ − 30cos²θ + 3 has zeros at
θ ≈ 30.6° and 70.1°. If the NO₂ pair angles in the principal frame sit near those zeros, the L=4
terms *would* be genuinely suppressed, and the q dependence factorises out of `Y`, so the
suppression would indeed apply at every q. This is a real possibility, not a hand-wave.

Note also that the paper's Fig. 3c is for **N₂O and symmetric-ish** cases; the stretched NO₂
geometry has different pair angles.

## What was ruled out

- **Not the `sph_harm` → `sph_harm_y` migration.** All three call sites were verified numerically
  identical before/after to 5.6e-16 under their exact broadcast shapes (see `how_to_run.md` §6).
- **Not a data/model inconsistency.** With the PDF model, `log_likelihood(truth) == 0` exactly and
  all 12 parameter perturbations lower it, so the forward model reproduces its own simulated data.
  The suppression affects data and model equally, which is precisely why it does not show up as a
  retrieval failure — it just removes information.

## What was not done

An independent re-derivation of `C_lmk` from paper Eq. 21 was attempted and abandoned: I misused
the internal helpers (`calculate_dists` expects a **2-D `[atoms, xyz]`** array, and
`rotate_to_principalI` expects `[N, atoms, xyz]`), got nonsense distances, and did not carry the
check through. **This is the missing piece of work.**

## Suggested investigation

1. Print the molecular-frame pair geometry and check the polar angles against the Y₄⁰ zeros:
   ```python
   mol = ex.rotate_to_principalI(mols[:, 0])      # [N, atoms, xyz] in, same out
   d   = ex.calculate_dists(mol[0])               # 2-D input! -> [pairs, (r, polar, azim)]
   # expect r = 1.35, 1.05, 2.213 for the stretched geometry
   ```
   If the polar angles land near 30.6° or 70.1°, the suppression is physical.
2. Evaluate `Y_4^{-k}` term by term for the three pairs and look for the cancellation explicitly.
   A physical cancellation will show large individual terms summing to near zero; a bug will show
   individually tiny terms.
3. Cross-check with a *different* geometry (e.g. the symmetric `NO2.xyz`, or CF2IBr in
   `NO2/XYZ/`). If L=4 is suppressed for every geometry, it is a bug.
4. Compare against the scipy backend — but note it is currently broken, see
   [004](004-calc-type-1-and-2-broken.md) and the broadcasting bug in
   [010](010-dead-and-broken-code-paths.md).

## Related

- [001](001-ston-signal-to-noise-unusable.md) — the L=4 S/N figures there are dominated by this
- [004](004-calc-type-1-and-2-broken.md) — the natural cross-check backend does not run
