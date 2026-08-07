# 010 — Six dead or broken code paths that raise on first use

**Severity** P3 individually (all off the main path), but collectively a trap: none of them fails
until you enable the feature, and several are reachable from documented parameters.
**Area** dead code
**Status** open

Grouped in one issue because the fix is the same decision each time: **repair or delete**. Nothing
here has ever run in this checkout.

## 10a. `calculate_coeffs_ensemble_scipy` — broadcasting bug

`modules/density_extraction.py:1452`, failing at `:1503`:

```
ValueError: operands could not be broadcast together with
            shapes (6,3,52,1,1) (4,3,52,1,1)
```

The leading axis is 6 (the number of fitted LMK) on one operand and 4 on the other. This blocks the
scipy ensemble backend, which is the natural independent cross-check for
[002](002-L4-coefficients-anomalously-small.md) — that is the real cost of leaving it broken.

Reproduce:
```python
ex.calculate_coeffs_ensemble_scipy(mols, w)   # mols from molecule_ensemble_generator
```

## 10b. `remove_global_offset` — two `NameError`s and a `TypeError`

`:2233`. Reached when `global_offset` is present in the parameter dict.

| Line | Problem |
|---|---|
| `:2239`, `:2242` | `fit_I0(...)` called without `self.` → `NameError` |
| `:2225` | `self.ensemble_generator(..., N=...)` — the generator takes no `N` keyword → `TypeError` |

See [006](006-measured-data-path-broken.md) for the `fit_I0` family and
[009](009-ensemble-grid-size-parameter-ignored.md) for the `N=` keyword.

## 10c. `simulate_error_data` — the `("data", …)` error model has never run

`:2676`. Selected by `simulate_error = ("data", (order, Wn))`, which **is advertised in
`README.md`'s parameter list**.

| Line | Problem |
|---|---|
| `:2692`, `:2695` | `fit_I0(...)` without `self.` → `NameError` |
| — | references `plot_folder`, which is a local of `simulate_data` (`:1010`), not in scope here |

Of the four documented error models, this makes two broken (`data` here, `StoN` unusable per
[001](001-ston-signal-to-noise-unusable.md)) and one shape-suspect (10f below).

## 10d. `compare_c_coeffs_scipy` — undefined locals

`:1134`. Reached when `compare_c_coeffs = True`. References `ensemble`, `weights` and
`input_data_coeffs` as if they were locals; they are not. Ironically this is the built-in
C++-vs-scipy validation routine — again relevant to
[002](002-L4-coefficients-anomalously-small.md).

## 10e. `save_results` / `get_results` — dead and broken

`:2396` and `:2413`. Never called anywhere (grep-verified). Both broken:

- `:2399` `self.get_fileName(np.prod(probs.shape[1]))` — passes an int positionally, so it binds to
  `folder_only`.
- `:2416` `self.get_fileName(1000)` — same.
- `:2408`, `:2435` read `data_params["perturb_range"]`, which `NO2/parameters.py` never defines.
  (It *is* used legitimately by `get_molecule_perturber` at `:3023-3064`, so the parameter is real —
  just absent from the NO₂ configuration.)

Given `save_emcee_backend` supersedes them, deletion is probably right.

## 10f. `constant_background` produces a 1-D `experimental_var`

`:1019`:

```python
self.experimental_var = np.ones(self.input_data_coeffs.shape[-1], dtype=float)
```

That is shape `(n_dom,)`, whereas the `constant_sigma` branch produces `(n_lmk, n_dom)`. It
broadcasts far enough to survive `simulate_data`, but `prune_data:1276` does
`self.experimental_var[:, dom_mask]`, which indexes the **q axis as if it were the LMK axis**. For
simulated data `data_or_sim` is False so that line is skipped and the bug hides; with imported data
it will slice the wrong axis or raise.

Prefer `constant_sigma` until this is fixed. Fix is presumably:

```python
self.experimental_var = np.ones_like(self.input_data_coeffs)
```

## 10g. Dead ADM block behind `and False`

`:1056`:

```python
if "sigma" not in error_type and False:
```

Permanently unreachable — roughly 25 lines of ADM handling inside `simulate_data`. Because of it,
`simulate_data` itself never touches the ADMs; `simulate_error_StoN` is the only live consumer.
Either restore the condition or delete the block; leaving `and False` in place obscures which code
is real.

## Suggested approach

Triage in this order:
1. **10a** and **10d** — repair, because they unblock the cross-check needed for
   [002](002-L4-coefficients-anomalously-small.md).
2. **10f** — repair; it is a two-line fix on a documented parameter.
3. **10c** — repair or remove from `README.md`'s parameter list; do not leave it advertised.
4. **10b**, **10e**, **10g** — delete unless there is a plan to use them. A CI job that merely
   imports every module and constructs `density_extraction` under each documented parameter
   combination would have caught all of these.
