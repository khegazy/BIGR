# 019 — The C-coefficient prefactor differs from paper Eq. 21 by an L-dependent factor

**Severity** P2 (self-consistent for simulate-and-fit; a real bias against measured data)
**Area** C coefficient calculation / conventions
**Status** open — needs a decision from the authors about which convention is intended

## The discrepancy

Paper Eq. 21:

```
H_lmk(q,R) = I * Re{ (-1)^k * 32*pi^3 * i^l/(2l+1)
                     * sum_{mu != nu} f_mu f_nu* j_l(q dR) Y_l^-k(theta, phi) }
```

Code (`modules/density_extraction.py:1629`, and the same expression in the scipy backend):

```python
c_prefactor = 1j**self.data_LMK[:,0]*8*np.pi**2/(2*self.data_LMK[:,0] + 1) \
              *np.sqrt(4*np.pi*(2*self.data_LMK[:,0] + 1))
```

The ratio is **L-dependent**:

| L | Eq. 21: 32π³/(2l+1) | code | ratio |
|---|---|---|---|
| 0 | 992.20 | 279.89 | 3.545 |
| 2 | 198.44 | 125.17 | 1.585 |
| 4 | 110.24 | 93.30 | 1.182 |
| 6 | 76.32 | 77.63 | 0.983 |

Analytically the ratio is `4π / sqrt(4π(2L+1)) = sqrt(4π/(2L+1))`, i.e. exactly the normalisation
between the semi-normalised (Racah) spherical harmonics `C_l^m = sqrt(4π/(2l+1)) Y_l^m` and the
orthonormal `Y_l^m`. The code evaluates `scipy.special.sph_harm_y`, which is orthonormal, and folds
`sqrt(4π(2L+1))` into the prefactor — so it is internally consistent, but it is using a different
`Y` normalisation than Eq. 21 as written.

## Why it matters, and why it has not shown up

Because the factor depends on L, it is **not** an overall scale: it reweights the different
anisotropy orders relative to one another by up to 3.6× between L = 0 and L = 6 (1.6× between
L = 2 and L = 6).

- **Simulated data:** harmless. Data and model pass through the same prefactor, so it cancels and
  `logL(truth) = 0` exactly.
- **Measured data:** *not* harmless. If imported C_lmk were extracted from experiment using Eq. 21's
  convention, fitting them with the code's convention biases the relative weighting of L, and the
  fit would absorb that into Θ. This is on the `data_fileName` path, i.e. the one that matters for
  real measurements.
- **Comparing to published M_lmk values** (paper Fig. 3c, 4b) requires knowing which convention
  those numbers are in.

## Also: the (-1)^k factor has no counterpart in the code

Eq. 21 carries `(-1)^k`; `c_prefactor` does not. **Currently harmless**, because every k in use is
even (`fit_bases` has k = 0, 2, 4 and the ADM files are `D{L}{K}` with even K), so `(-1)^k = +1`
always. It would matter the moment an odd k is fitted. Worth adding either the factor or an
assertion that k is even.

## Recommended action

Not a mechanical fix — it needs the authors to state which `Y` normalisation Eq. 21 assumes. Then:

1. If Eq. 21 means orthonormal `Y_l^m`, drop the `sqrt(4π(2L+1))` from `c_prefactor` and use
   `32π³` in place of `8π²`.
2. If the code's convention is intended, add a comment saying so and note the relationship to
   Eq. 21, so nobody "fixes" it later.
3. Either way, document which convention an imported `fit_coeffs_dataLMKindex-{i}` dataset is
   expected to be in — see the measured-data section of `how_to_run.md`.
4. Add `(-1)**k` to the prefactor, or assert `np.all(fit_bases[:,2] % 2 == 0)`.
