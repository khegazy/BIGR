# 004 — `calc_type` 1 and 2 both crash, and `calc_type` is silently ignored when multiprocessing is on

**Severity** P1 (two of three documented backends unusable)
**Area** spherical Bessel backends
**Status** open

`README.md` advertises three spherical-Bessel implementations selected by `calc_type`:

```
0 <- C++ implementation (Recommended, but cannot include very low q)
1 <- Scipy implementation (Slowest but correct for all q values)
2 <- Optimized Python implementation (Slower than 0 with the same errors)
```

Only `calc_type = 0` works. This matters because option 1 is the documented fallback when the C++
extension will not compile, and it is the only backend that is correct at low q.

## 4a. `calc_type = 1` crashes with an arity mismatch

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

**Fix.** Give the scipy version a compatible signature and ignore the second argument, matching the
C++ wrapper's `calculate_even_only(x, N_qbins=-1)`:

```python
def numpy_jn(x, N_qbins=-1):
    return sp.special.spherical_jn(self.data_Lcalc, x)
```

(Guarding the `compare_spherical_bessel_scipy` call instead would also work, but that comparison is
meaningless for the scipy backend anyway — it would be comparing scipy against itself.)

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

Since option 2 is advertised as "same errors as 0 but slower", and 0 works, the cheapest honest fix
may be to **delete the branch** and drop it from `README.md` rather than repair it.

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
