# 016 — Ensemble discretisation error dominates the PDF likelihood and makes it unsamplable

**Severity** **P1 — the most consequential issue found.** It silently corrupts the PDF (`P^(N)`)
likelihood surface, which is the paper's central model.
**Area** `molecule_ensemble_generator` / likelihood
**Status** open. A validated fix exists (Gauss–Hermite quadrature, below) that is simultaneously
~30× more accurate and ~2× cheaper than the current default, but it has not been run through a full
retrieval yet.

## Summary

`molecule_ensemble_generator` approximates the Gaussian integral of paper Eq. 4 by a discrete sum on
an outer-product grid. The quadrature is far too coarse at any affordable grid size, and its error —
**not the data** — dominates how the likelihood varies with Θ. Consequences:

- It **contributes to** a rugged, multi-modal likelihood surface on sub-percent scales in Θ, which
  emcee cannot sample (acceptance ≈ 2.7 %, and τ grows linearly with chain length instead of
  plateauing, so `iteration > 100·τ` is never satisfiable). But see the correction below: the
  quadrature is *not* the sole cause of the ruggedness.
- Any σ calibrated by measuring likelihood curvature at a coarse grid is **wrong**, because the
  curvature being measured is quadrature noise.
- The error is largest for small Θ perturbations, i.e. exactly the regime that sets the reported
  resolution σ^Θ.

## Evidence

`logL` at fixed offsets in ⟨NO⁽¹⁾⟩ does not converge as the grid is refined (asymmetric NO₂,
q ≤ 10 Å⁻¹, `constant_sigma = 0.163`, grid span ±7σ):

| N | spacing | s/θ | ΔlogL @ +0.2% | @ +0.5% | @ +1.0% |
|---|---|---|---|---|---|
| 11 | 1.40σ | 0.03 | −7.885 | −1.468 | −9.597 |
| 19 | 0.78σ | 0.16 | −2.718 | −5.822 | −8.029 |
| 31 | 0.47σ | 0.69 | −0.499 | −0.774 | −0.090 |
| 51 | 0.28σ | 3.08 | −0.137 | −0.202 | −0.500 |
| 71 | 0.20σ | 9.46 | −0.061 | −0.428 | −0.075 |
| 91 | 0.16σ | 20.24 | −0.012 | −0.065 | −0.050 |

Successive-N relative changes are 50 % to 8853 %. The values shrink toward zero, so the *true*
sensitivity at these offsets is tiny and everything above it at small N is numerical error. At
N = 11 the error at +2 % is roughly **25 000 %** of the converged value.

A scan of ΔlogL over ±2 % in ⟨NO⁽¹⁾⟩ at N = 11 illustrates the ruggedness directly — it is not
even monotonic:

```
 -5.0%  -0.88     +0.5%   -1.47
 -2.0%  -3.66     +1.0%   -9.60
 -1.0% -12.18     +2.0%  -35.99
 -0.5%  -0.47     +5.0%   -2.98
  0.0%   0.00    +10.0%   -3.23
```

At large offsets the quadrature does converge (ΔlogL at +10 % settles near −2.5 for every
N ≥ 31), which confirms the problem is specifically the small-Θ behaviour.

## Correction: how much of the ruggedness is actually the quadrature?

**Less than this issue originally claimed.** The delta model (`density_model: "delta"`,
`single_molecule_generator`) evaluates a single geometry with weight 1 — there is no ensemble sum, so
**no quadrature error is possible**. Yet with delta-generated data (so `logL(truth) == 0` exactly)
its likelihood is *also* non-monotonic near truth:

```
offset in d1    dlogL          offset in d1    dlogL
   -2.00%     -2370.3            +0.10%        -65.0
   -1.00%      -397.1            +0.20%        -42.3   <- better than +0.10%
   -0.50%      -365.7            +0.50%        -44.6
   -0.20%      -152.7            +1.00%       -123.2
   -0.10%      -382.9  <- worse  +2.00%      -5053.8
    0.00%         0.0               than -0.20%
```

**Direct evidence that the grid nevertheless drives the sampling difficulty:** the emcee acceptance
fraction, on otherwise identical configurations (σ = 0.163, q ≤ 5 Å⁻¹, 32 walkers, asymmetric NO₂):

| `ENSEMBLE_GRID_N` | acceptance fraction |
|---|---|
| 11 | **0.027** (pathological) |
| 19 (shipped default) | **0.230** (healthy; emcee's useful range is ~0.2–0.5) |

An 8.5× change in acceptance from the integration grid alone — nothing else differed — confirms the
grid is what broke sampling at N = 11, even though it is not the *only* source of structure in the
surface.

So a rugged surface is not by itself evidence of quadrature error. Two sources have to be separated:

1. **Genuine structure in the exact likelihood.** The quadratic fit above gives 1σ ≈ 0.53 mÅ on d₁,
   so these ±0.1 % offsets are ≈2.5σ out. Beyond ~2σ, a χ² built from oscillatory fringe data
   (`j_l(q·ΔR)` integrated to q = 10 Å⁻¹) legitimately develops side lobes and local minima — the
   usual fringe/phase ambiguity of interferometric fitting. This structure is **real** and is present
   even with a perfect integrator.
2. **Quadrature error.** Distinguished from (1) by one decisive property: **it must shrink under grid
   refinement.** Genuine physical structure cannot depend on the integration grid, whereas the tables
   above show logL at a *fixed* Θ changing by up to 25 000 % between N = 11 and N = 91, and the
   fixed-data bias at Θ_truth being −448 at N = 19. That evidence is unaffected by (1) and still
   stands.

**What changes:** the claim "the quadrature error makes the surface unsamplable" is too strong. Some
of the sampling difficulty is intrinsic. So fixing the quadrature is necessary for an unbiased
forward model (consequence (2) below) but may **not** on its own make the posterior easy to sample —
better initialisation near the global mode, tempering, or a sampler suited to multimodal targets may
be needed regardless. This is consistent with the paper's own design, which locates Θ\* with a
separate mode search (`modules/mode_search.py`) rather than reading it off the MCMC marginals.

The delta scan above was run through a hand-built configuration rather than the normal parameter
path (see the note in [009](009-ensemble-grid-size-parameter-ignored.md) about
`sim_thetas[:-1]`), so it is worth repeating through a supported route before leaning on it hard.

## Why it is not self-cancelling

Simulated data and the fitted model both pass through `molecule_ensemble_generator` with the same
grid, so it is tempting to assume the discretisation error cancels. **It cancels only exactly at
Θ_truth.** Away from truth the grid slides through the Gaussian and the quadrature error changes,
so the error contributes structure to the likelihood *as a function of Θ* — precisely the thing
being inferred. This is why `logL(truth) == 0` exactly (a genuine and reassuring check) while the
surface around it is garbage.

Note also that reducing σ does **not** improve the ratio: the quadrature error is a bias in the model
prediction and enters χ² as `(model_error/σ)²`, so it scales with σ exactly as the statistical signal
does. Grid resolution is the only lever.

## Cause

`modules/NO2.py` builds each 1-D grid as `np.linspace(x - std_*sigma, x + std_*sigma, N)` with
`std_ = 7` (now the constant `ENSEMBLE_GRID_SPAN`). The spacing is

```
spacing = 2 * span * sigma / (N - 1)
```

With span 7 and N = 11 that is 1.40σ — hopelessly coarse. The uniform grid is simply the wrong
quadrature rule for a Gaussian-weighted integral of an oscillatory function.

### A tempting mitigation that does NOT work: shrinking the span

It looks obvious that ±7σ wastes points, since the outermost node carries weight
`exp(-7²/2) ≈ 4e-11`. **This is a trap, and I fell into it before testing properly.** The integrand
is not slowly varying: it contains `j_l(q·ΔR)` with `q·σ` of order radians, so the tail nodes still
carry real signal. Measured against the exact integral `E[exp(i k x)] = exp(-k²s²/2)` for
`x ~ N(0, s)` at N = 19:

| k·s | span 7 relative error | span 3 relative error |
|---|---|---|
| 0.3 rad | 4e-4 | 7e-3 |
| 1.8 rad | 1e-4 | 5e-3 |
| **3.0 rad** | **3e-4** | **7e-2  (200x worse)** |

So **span 7 — the original value — is correct**, and reducing it makes things worse despite the finer
spacing. Do not change it.

(For the record: an earlier version of this issue recommended span 3 on the basis of a comparison run
through the full pipeline. That test was confounded — each grid setting regenerated its own simulated
data with its own quadrature error, so it compared schemes fitting *different* data rather than
measuring model accuracy. The exact-integral test above is the valid one.)

## Two distinct consequences, which must not be conflated

Holding the **data fixed** at a dense reference grid (N = 91) and varying only the **model** isolates
the model's own accuracy. `constant_sigma = 0.05`, q ≤ 10 Å⁻¹, 6 LMK × 118 q = 708 data points:

| model | geometries | logL(truth) | +0.2% | +0.5% | +2% | +5% |
|---|---|---|---|---|---|---|
| uniform N = 91 sp7 (same as data) | 753 571 | 0.000 | −0.132 | −0.695 | −1.505 | −6.238 |
| **uniform N = 19 sp7 (shipped default)** | 6 859 | **−448.4** | −265 | −227 | −259 | −332 |
| uniform N = 11 sp7 | 1 331 | −335.6 | −668 | −445 | −1304 | −258 |
| Gauss–Hermite 15 | 3 375 | −36.5 | −8.9 | −5.1 | −24.2 | −17.0 |
| Gauss–Hermite 21 | 9 261 | −63.3 | −33.0 | −29.7 | −28.7 | −24.5 |

The top row is the true likelihood: smooth and monotonic, as it should be. Read the rest as two
separate problems.

**(1) Self-consistent simulated data — a sampling problem, not a bias.** In normal operation the
simulated data and the model use the *same* grid, so the quadrature error cancels exactly at Θ_truth
(`logL(truth) == 0`) and the retrieval is self-consistent. What survives is the *variation* of the
error with Θ, which is the ruggedness documented above. The practical damage is that emcee cannot
sample the surface.

**(2) Real measured data — a genuine systematic bias.** Against data that is *not* generated by the
same grid, the model is simply wrong: at N = 19 the discrepancy at Θ_truth is 448 in logL, i.e. the
forward model is many σ away from a correct evaluation of Eq. 4. Fitting real data with it would give
a systematically biased Θ, and the fit could not distinguish that bias from a structural signal.
**This is the more serious consequence and it applies to the shipped default, not just to my
reduced grid.**

Two important caveats on how strongly to read (2):

- The logL bias scales as 1/σ². At `constant_sigma = 0.163` the same model error gives ≈ −42 rather
  than −448. So the *number* is σ-dependent; what is σ-independent is that the model error greatly
  exceeds the statistical error whenever the data is reasonably precise.
- This was measured at one geometry, one σ and one q range. Before drawing conclusions about
  published results it should be repeated across the paper's actual configurations.

## The fix that does work: Gauss–Hermite quadrature

The integral is genuinely Gaussian-weighted, so Gauss–Hermite is the textbook rule.
`numpy.polynomial.hermite_e.hermegauss(n)` gives nodes in units of σ and weights for `exp(-x²/2)`.
Verified against the same exact integral:

| scheme | geometries (3 dof) | rel. error @ k·s = 3 rad |
|---|---|---|
| uniform N = 11, span 7 (**avoid**) | 1 331 | 4e-1 |
| uniform N = 19, span 7 (current default) | 6 859 | 3e-4 |
| Gauss–Hermite n = 7 | 343 | 3e+0 (fails) |
| **Gauss–Hermite n = 15** | **3 375** | **1e-5** |

On the exact scalar integral, Gauss–Hermite with 15 nodes is ~30× more accurate than the current
default while using half the geometries (15³ = 3375 versus 19³ = 6859). Note n = 7 is *not* enough —
the oscillation needs the higher-order rule — so the naive "few nodes suffice" intuition fails here.

**But in the full pipeline it helps without solving the problem.** In the fixed-data table above,
GH 15 cuts the bias at Θ_truth from −448 to −36.5 (12× better, at half the cost) — a real
improvement, but still far from the reference. And GH 21 is *worse* than GH 15, so it is not
converging monotonically in n either.

The likely reason is that Gauss–Hermite is optimal for (polynomial × Gaussian), and this integrand is
neither: the O–O distance depends nonlinearly on (d₁, d₂, ∠ONO), the angle enters trigonometrically,
and `j_l(q·ΔR)` oscillates. So the Gaussian-weight part is handled well while the rest is not. A
correct fix probably needs either many more nodes, an adaptive/nested rule with an error estimate, or
a change of variables that makes the integrand smoother in the integration coordinates.

A drop-in replacement, keeping the existing `(molecules, probs)` contract:

```python
from numpy.polynomial.hermite_e import hermegauss

def molecule_ensemble_generator(thetas, n=15):
    x, w = hermegauss(n)                      # nodes in units of sigma
    w = w/np.sum(w)
    d1, s1, d2, s2, a, sa = (thetas[:, i] for i in range(6))
    i1, i2, i3 = (v.ravel() for v in np.meshgrid(*[np.arange(n)]*3, indexing="ij"))
    th = np.stack([d1[:, None] + s1[:, None]*x[i1][None, :],
                   d2[:, None] + s2[:, None]*x[i2][None, :],
                   a [:, None] + sa[:, None]*x[i3][None, :]], axis=-1)
    probs = np.tile((w[i1]*w[i2]*w[i3])[None, :], (thetas.shape[0], 1))
    return theta_to_cartesian_ensemble(th), probs/np.sum(probs, -1, keepdims=True)
```

Because the nodes are fixed in units of σ *relative to the mean*, the quadrature error also becomes a
smooth function of Θ rather than oscillating as a uniform grid slides through the Gaussian — which is
what produced the ruggedness in the first place. **This change was validated on the exact integral but
has NOT been run through the full retrieval**; do that before adopting it.

## Other things worth doing

1. **Sparse grids / Smolyak** if the degree-of-freedom count grows. The full outer product is
   exponential in the number of dof, which will get much worse for molecules larger than NO₂.
2. **Report the discretisation error at startup.** Evaluate the likelihood at `n` and `2n` once and
   warn if it moves by more than a tolerance. Cheap, and it would have caught all of this immediately.

## Impact on published results

Worth checking before it matters: the paper's runs used span = 7 with N = 19 (the committed default),
which the table above puts at ~82 % deviation from the converged reference at small offsets. That
should be re-examined against a Gauss–Hermite or high-N reference to confirm the quoted σ^Θ
resolutions are statistical rather than quadrature-limited. I have not been able to determine this
either way.

## Related

- [009](009-ensemble-grid-size-parameter-ignored.md) — how the grid size is configured, and the
  inert `51` in `sim_thetas`. The intended 51 would have given 0.28σ spacing at span 7, consistent
  with someone having noticed accuracy mattered.
- [002](002-L4-coefficients-anomalously-small.md) — separate issue, but a working scipy cross-check
  would help here too.
- `how_to_run.md` §7 — the cost table and the fast-parameter rationale.
