# 004 — `calc_type` 1 and 2 both crash, and `calc_type` is silently ignored when multiprocessing is on

**Severity** ~~P1~~ P3
**Area** spherical Bessel backends
**Status** **4a and 4c fixed** 2026-08-08. 4b (`calc_type = 2`) partially repaired but still
broken — recommend deleting the branch rather than finishing it; see below.

`README.md` advertises three spherical-Bessel implementations selected by `calc_type`:

```
0 <- C++ implementation (Recommended, but cannot include very low q)
1 <- Scipy implementation (Slowest but correct for all q values)
2 <- Optimized Python implementation (Slower than 0 with the same errors)
```

Originally only `calc_type = 0` worked. **`calc_type = 1` now works** and is a genuinely independent
backend (it uses `scipy.special.spherical_jn` rather than the C++ recursion); the two agree to
~1e-10, which is checked by `scripts/test_physics.py`. That independence is what confirmed the C++
recursion after the centre-of-mass fix in [002](002-L4-coefficients-anomalously-small.md).
`calc_type = 2` remains broken.

## 4a. `calc_type = 1` crashes with an arity mismatch — ✅ FIXED

`setup_calculations` defines the scipy evaluator taking **one** argument
(`modules/density_extraction.py:2527`):

```python
def numpy_jn(x):
    return sp.special.spherical_jn(self.data_Lcalc, x)
```

but `__init__` unconditionally calls `compare_spherical_bessel_scipy`, which invokes it with
**two** (`:680`):

```python
j_check = self.spherical_j(self.dom, len(self.dom))
```

→ `TypeError: numpy_jn() takes 1 positional argument but 2 were given`, during construction.

**Fixed 2026-08-08.** `numpy_jn` now takes the same `(x, N_qbins=-1)` signature as the C++ wrapper
*and* returns one row per even order indexed by `l/2`, so both backends share a single convention.
That second part matters: without it `compare_spherical_bessel_scipy`'s `n//2` indexing silently
reads the wrong row for `calc_type=1`. `calculate_coeffs_ensemble_scipy` also had to be fixed
([010a](010-dead-and-broken-code-paths.md)) before `calc_type=1` was usable end to end.

`calc_type=1` is now a genuinely independent backend — it uses `scipy.special.spherical_jn` rather
than the C++ recursion — and `scripts/test_physics.py` checks all three paths agree to 5.6e-11.
That independence is what confirmed the C++ recursion after the centre-of-mass fix.

## 4b. `calc_type = 2` is dead code — two `NameError`s

In the `calc_type == 2` branch (`:2534`):

- `:2538` and `:2542` reference an undefined name `n`:
  ```python
  N = (np.unique(n)/2).astype(int)
  ...
  for nn in np.unique(n):
  ```
  Almost certainly meant to be `self.data_Lcalc` (the L values), given the surrounding recursion
  over even orders.
- `:2629` is missing `self.`:
  ```python
  self.calculate_coeffs = calculate_coeffs_ensemble_scipy   # NameError: bare name
  ```
  should be `self.calculate_coeffs_ensemble_scipy`.

**Partially repaired 2026-08-08, still does not run.** Three defects were fixed — `n` is now defined
as `self.data_LMK[:,0]` (its uses `np.unique(n)` and `np.sum(nn == n)` make that unambiguous), the
missing `self.` on the `calculate_coeffs` assignment, and the same one-argument
`calculate_even_only` arity mismatch that `calc_type = 1` had. It then fails deeper in, inside the
pure-Python Bessel evaluator:

```
IndexError: too many indices for array: array is 1-dimensional, but 4 were indexed
  density_extraction.py:2683, in calculate_even_only
```

So the branch has structural shape assumptions that do not match how `spherical_j` is called, not
just typos.

**Recommendation: delete it.** `README.md` advertises option 2 as having "the same errors as 0" while
being slower, so it offers nothing over `calc_type = 0`, and there are now *two* working and
mutually-agreeing backends (0 and 1) — which is all the cross-checking value option 2 could have
provided. Repairing it would mean debugging a hand-rolled 1/x expansion of the spherical Bessel
functions for no benefit. If it is deleted, drop it from `README.md`'s `calc_type` list too.

## 4c. `calc_type` is silently overridden by `multiprocessing`

`:2501`:

```python
if self.data_params["calc_type"] == 0 or self.do_multiprocessing:
```

`do_multiprocessing` is true whenever `multiprocessing > 1`. So with the shipped default
`multiprocessing = 10`, setting `calc_type = 1` **still takes the C++ branch** — the very
combination someone would try when the C++ extension is the suspect. Nothing warns them.

**Fix.** Separate the two concerns: pick the evaluator from `calc_type` alone, and pick
`calculate_coeffs` from `do_multiprocessing` alone. If a scipy multiprocessing path genuinely does
not exist, raise a clear error instead of silently switching backends:

```python
if self.do_multiprocessing and self.data_params["calc_type"] != 0:
    raise ValueError("multiprocessing requires calc_type=0; got {}".format(
        self.data_params["calc_type"]))
```

## Reproduce

```bash
cd NO2
# 4a + 4c: must set BOTH to reach the scipy branch at all
MPLBACKEND=Agg ../.venv/bin/python - <<'PY'
from parameters import get_parameters
from modules.density_extraction import density_extraction
from modules.NO2 import *
p = get_parameters(); p["calc_type"] = 1; p["multiprocessing"] = 0
density_extraction(p, get_molecule_init_geo, get_scattering_amplitudes,
    log_prior=log_prior_3dof_gauss, density_generator=molecule_ensemble_generator,
    ensemble_generator=molecule_ensemble_generator, get_ADMs=get_ADMs)
PY
```

## Related

- [010](010-dead-and-broken-code-paths.md) — `calculate_coeffs_ensemble_scipy` also has a
  broadcasting bug, so even after fixing 4a the scipy *ensemble* path will not run
- [002](002-L4-coefficients-anomalously-small.md) — a working scipy backend is the natural
  cross-check for that anomaly, which is why this is more than cosmetic
