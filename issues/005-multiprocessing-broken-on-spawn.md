# 005 — `multiprocessing > 1` cannot work on macOS or Windows (spawn cannot pickle a closure)

**Severity** P1 (the only parallelism in the package is unusable off Linux)
**Area** performance
**Status** open — worked around by forcing `multiprocessing = 0`

## Symptom

With `multiprocessing > 1` on macOS (or Windows), the first likelihood evaluation raises a
pickling error such as:

```
AttributeError: Can't pickle local object 'density_extraction.setup_calculations.<locals>.calculate_even_only'
```

## Cause

`calculate_c_ensemble_multiProc` (`modules/density_extraction.py:1525`) launches workers with a
**bound method** as the target (`:1553`):

```python
mp.Process(target=self.calculate_c_ensemble_multiProc_helper, args=(...))
```

Pickling a bound method pickles `self`, and `self` holds objects that cannot be pickled:

- `self.spherical_j` — assigned at `:2516` to `calculate_even_only`, a **closure defined inside
  `setup_calculations`**. Local functions are not picklable at all.
- `self.mp_manager` — a `multiprocessing.SyncManager` instance.
- `self.scat_amps_interp` — scipy `interp1d` objects (these do pickle, but add cost).

Python ≥ 3.8 defaults to the **spawn** start method on macOS, which pickles the target. On Linux the
default is **fork**, which does not, so the code works there — which is why this was never seen on
the cluster.

## Impact

Two separate costs:

1. On macOS/Windows the package is effectively single-core. For the PDF model that is the
   difference between ~0.56 s and ~0.1 s per MCMC step on a 10-core machine.
2. `setup_calculations:2501` routes to the C++ backend whenever `do_multiprocessing` is set, so a
   user who tries `multiprocessing = 10` to go faster also silently loses their `calc_type` choice
   — see [004c](004-calc-type-1-and-2-broken.md).

## Suggested fix

Make the worker target a **module-level function** taking only picklable arguments, rather than a
bound method. The evaluator needs to be reconstructible in the child, e.g. pass `calc_type` and
have the child call a module-level factory:

```python
# module level, not a closure
def _spherical_j_from_calc_type(calc_type, data_Lcalc):
    ...

def _coeffs_worker(chunk_id, ensemble, weights, calc_type, data_Lcalc,
                   dist_inds, sms_scat_amps, dom, lmk, return_dict):
    spherical_j = _spherical_j_from_calc_type(calc_type, data_Lcalc)
    return_dict[chunk_id] = _calculate_coeffs(...)
```

Cheaper alternatives, in increasing order of desirability:

1. **Document and enforce.** Detect the start method and raise a clear error rather than a cryptic
   pickling traceback:
   ```python
   if self.do_multiprocessing and mp.get_start_method() != "fork":
       raise RuntimeError(
           "multiprocessing requires the 'fork' start method; set multiprocessing=0. "
           "See issues/005.")
   ```
2. **Use `concurrent.futures` with a thread pool.** The inner loop is numpy/C++ heavy and releases
   the GIL, so threads may parallelise adequately with **no pickling at all** — likely the best
   effort-to-benefit ratio here.
3. Force `mp.set_start_method("fork")`. **Not recommended**: fork is unsafe in a process with
   threads, and macOS Accelerate-linked numpy uses threads, so this can deadlock or crash.

Note the arrays involved are large (`[3, n_q, n_walkers, N³]`), so per-call pickling of the payload
would itself be expensive — another argument for threads over processes.

## Workaround in place

`NO2/parameters.py:95-102` sets `multiprocessing = 0` for both density models, with a comment. See
`how_to_run.md` §7.
