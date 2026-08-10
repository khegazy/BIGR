# 002 — L=4 C coefficients are ~10⁻⁶ while L=2 and L=6 are ~10⁻², unexplained

**Severity** P1 (it corrupted every C coefficient)
**Area** C coefficient calculation
**Status** **RESOLVED 2026-08-08.** Root cause was a centre-of-mass bug in
`rotate_to_principalI`/`rotate_to_principalI_ensemble` that scaled every molecule by
1/total_mass — 46× for NO₂ — so all pairwise distances were 46× too small. That drove the C++
Bessel upward recursion into its unstable regime, producing erratic values for the higher orders.

**Fix:** the division by the total mass belongs to the mass-weighted sum, not to the coordinates:

```python
# wrong: r'' = (r - sum_j m_j r_j)/M  -> scales all coordinates by 1/M
molecules -= np.sum(molecules*self.mass[:,0], axis=-2, keepdims=True)
molecules /= np.sum(self.mass)

# right
molecules = molecules \
    - np.sum(molecules*self.mass[:,0], axis=-2, keepdims=True)/np.sum(self.mass)
```

**After the fix:** rotation preserves pairwise distances exactly; C₄₀₀ (3.74) > C₆₀₀ (2.26) as
required; L = 2/4/6 come out 11.0 / 4.24 / 2.35 so M₄₀ₖ and M₆₀ₖ are ~2.6× and ~4.7× smaller than
M₂₀ₖ, matching paper Fig. 3c's ×2 and ×4 scale factors; and the C++, scipy-combination and
scipy-Bessel backends agree to 5.6e-11. Guarded by `scripts/test_physics.py`.

This also retracts [016](016-ensemble-quadrature-error-dominates-likelihood.md), which was
measuring the same artifact.

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

## Kareem's physical constraint (2026-08-08): C₄₀₀ must exceed C₆₀₀

This settles the "is it physical?" question — it is not. And it sharpens the symptom, because the
code is wrong in *both* directions:

| LMK | code rms | expected ordering |
|---|---|---|
| [2 0 0] | 0.0199 | |
| [2 0 2] | 0.0402 | |
| **[4 0 0]** | **1.31e-06** | should be **larger** than [6 0 0] |
| [4 0 2] | 1.57e-06 | |
| [4 0 4] | 1.62e-05 | |
| **[6 0 0]** | **0.2376** | should be **smaller** than [4 0 0] — and it is 12× *larger* than [2 0 0] |

So L=4 is far too small **and** L=6 is far too large. At the qΔR values probed, |C_L| should fall
with L (j_l is strongly suppressed for x ≲ l). **"L=4 too small, L=6 too large" is the signature of
the two orders being swapped or mis-indexed**, not of any smooth error.

## Verified correct — these are NOT the cause

Each of these was checked and can be excluded:

- **The C++ even-order recurrence.** Derived it algebraically: eliminating the odd order from
  `j_{n+1} = ((2n+1)/x)j_n − j_{n−1}` gives coefficients `(2n+1)(2n+3)`, `−(4n+2)/(2n−1)`,
  `−(2n+3)/(2n−1)`, which are exactly `recursive_coefficients` in
  `cpp_extensions/src/c_calc_extensions.cpp:84`.
- **The `jn` array layout and index arithmetic.** `spherical_j` writes j₀, j₂, j₄, j₆ at strides
  0…3 (`:145`, `:152`, `:157`) and `calculate_c` reads them with `j_idx_shift = (l[l_idx]/2)*…`
  (`:61`). Self-consistent.
- **ctypes integer width.** `modules/c_calc_extensions.py:115` does `lmk.astype(np.int32)`, so the
  int64 `data_LMK` is converted properly. No reinterpretation bug.
- **The `c_prefactor`.** `i^L` is real for all even L, so `c_prefactor_imag = 0` and the C++ uses
  only `Re(Ylk)` — that is the intended `Re{…}` of Eq. 21, not a dropped term.
- **The `sph_harm` → `sph_harm_y` migration** (verified identical to 5.6e-16 under the real
  broadcast shapes).

## RETRACTION: the Bessel functions are fine, and `keep_inds` must stay commented out

An earlier version of this issue claimed (a) that `[keep_inds]` was commented out in error and
should be restored, and (b) that this made `compare_spherical_bessel_scipy` compare misaligned rows
so its "Passed" was worthless. **Both claims are wrong.** Reading
`modules/c_calc_extensions.py:30-63` settles it:

```
Returns
-------
result : ... [l_max/2+1, d1, ..., dN]
    The even spherical bessel functions from 0 to the maximum l in lmk.
```

`spherical_j` returns **one row per even order from 0 to l_max**, indexed by `order/2` — *not* one
row per entry of `lmk`. For L = 2,2,4,4,4,6 that is **4 rows** (j₀, j₂, j₄, j₆), not 7.

Consequences:

1. **`keep_inds` is the wrong length by construction.** It is built with `len(lmk)` = 7 entries to
   mask the padded *list*, but the array it would index has 4 rows. Uncommenting `[keep_inds]` would
   raise or mis-select. It is almost certainly commented out *because* it broke. The whole
   padding + `keep_inds` block at `density_extraction.py:2515-2524` is vestigial: `spherical_j`
   already returns every even order up to l_max regardless of which appear in `lmk`, so no padding
   was ever needed. **Recommended action: delete the padding and `keep_inds` entirely**, keeping
   `lmk = self.data_LMK[:,0]`.
2. **`compare_spherical_bessel_scipy` is correct.** It uses `n_idx = n//2` and indexes
   `j_check[n_idx,:]` (`density_extraction.py:680-690`) — the right convention. So its
   `L = 2/4/6 ... Passed` at ~1e-14 **is** valid evidence, and `plots/check_jl4_calculation.png` is
   trustworthy.

**So the C++ spherical Bessel functions are verified correct, and so is their `l/2` mapping in
`calculate_c` (`c_calc_extensions.cpp:61`).** The L=4 problem is therefore *not* in the Bessel
functions or their indexing, and hypothesis H1 (a Bessel row/order swap) is **eliminated**.

That leaves the remaining factors in Eq. 21 as the candidates: the spherical harmonics `Ylk`, the
molecular-frame angles (θ, φ) feeding them, the pairing of `dist_sms_scat_amps` with the distance
axis, or the prefactor. The prefactor can be checked by hand and is *not* the cause — for L = 2/4/6
it evaluates to 125.2 / 93.3 / 77.6, all the same order.

## Why nothing caught this: every independent cross-check is broken

There is currently **no working second opinion** on the C coefficients anywhere in the package:

| intended cross-check | status |
|---|---|
| `compare_c_coeffs_scipy` — the built-in C++-vs-scipy test for precisely this | undefined locals ([010d](010-dead-and-broken-code-paths.md)) |
| `calculate_coeffs_ensemble_scipy` — the independent backend | broadcasting bug, `(6,3,52,1,1)` vs `(4,3,52,1,1)` ([010a](010-dead-and-broken-code-paths.md)) |
| `calc_type = 1` (scipy) | arity bug at `:680` ([004](004-calc-type-1-and-2-broken.md)) |
| `calc_type = 2` (python) | two `NameError`s ([004](004-calc-type-1-and-2-broken.md)) |
| `compare_spherical_bessel_scipy` | **works correctly** (uses `n//2`); it validates j_l but says nothing about the rest of Eq. 21 |

Fixing any one of these is worth more than further inspection by eye.

## Ranked hypotheses

1. ~~**H1 — Bessel row/order misalignment.**~~ **ELIMINATED** — see the retraction above. The
   Bessel values and their `l/2` mapping are both verified correct.
2. **H2 — `Ylk` axis ordering.** In `calculate_coeffs_ensemble_cpp:1636` `Ylk` is built from
   `data_Lcalc` shaped `(n_lmk,1,1,1)` and transposed `(0,2,3,1)` at
   `c_calc_extensions.py:120` before being passed. If the pairwise-distance axis ends up misaligned
   against `scat_amps` and `x`, each term pairs with the wrong ΔR.
3. **H3 — the molecular-frame rotation.** `rotate_to_principalI(mols[:,0])` returned shape
   `(3,3,3)` from a `(1,3,3)` input, with values ~0.1–0.3 rather than atom coordinates. Either its
   signature is not `[N, atoms, xyz]` or it returns something else. If the frame is wrong, the
   (θ,φ) feeding `Y_l^{−k}` are wrong. This would not single out L=4 on its own, but it needs
   independent verification. **Note:** `calculate_dists` takes a **2-D `[atoms, xyz]`** array —
   passing 3-D silently produces garbage, which caught me out twice.
4. **H4 — genuine physical cancellation.** Considered ruled out: it would have to suppress
   Y₄⁰, Y₄⁻² and Y₄⁻⁴ simultaneously by ~10⁴, and Kareem's expectation contradicts it.

## Recommended actions, cheapest first

1. **Restore `[keep_inds]`** (one line) so `check_jl*` becomes meaningful, then re-run and see
   whether L=4 still "passes".
2. **Instrument the C++ directly**: for one fixed `x`, dump `jn[l/2]` for each requested L and
   compare against `scipy.special.spherical_jn`. This settles H1 definitively in a few lines.
3. **Fix `calculate_coeffs_ensemble_scipy`** ([010a](010-dead-and-broken-code-paths.md)) to obtain a
   working independent backend, then compare per-LMK against the C++ path.
4. **Fix `compare_c_coeffs_scipy`** ([010d](010-dead-and-broken-code-paths.md)) — it exists for
   exactly this purpose.
5. **Verify `rotate_to_principalI`'s contract** and add a test asserting that pairwise distances are
   preserved by the rotation (a rotation cannot change them — a one-line invariant that would catch
   H3 immediately).
6. **Check the prefactor convention.** The code uses `i^L·8π²/(2L+1)·√(4π(2L+1))`
   (`density_extraction.py:1629`) while paper Eq. 21 has `32π³i^l/(2l+1)`. These may agree once the
   `Y` normalisation is folded in, but it should be confirmed rather than assumed.

## Status of my own attempts

I tried three times to recompute C_lmk independently and **failed each time** for harness reasons,
not physics: `calculate_dists` needs a 2-D array, `rotate_to_principalI` does not return what I
assumed, and my hand-rolled principal-axis rotation produced pairwise distances scaled by 1/45.99
(suspiciously exactly the total molecular mass, so probably a centre-of-mass/normalisation slip on
my part). **No numerical conclusion should be drawn from those attempts**; the ordering argument and
the `keep_inds` finding above stand on their own.

## What was ruled out

- **Not the `sph_harm` → `sph_harm_y` migration.** All three call sites were verified numerically
  identical before/after to 5.6e-16 under their exact broadcast shapes (see `CHANGELOG.md`).
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
