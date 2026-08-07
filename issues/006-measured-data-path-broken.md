# 006 — The measured-data path crashes: `self.fig_I0` does not exist

**Severity** P1 (the package cannot analyse real data out of the box)
**Area** real data / intensity fit
**Status** open — one-character fix, already flagged by a `TODO` in the source

## Symptom

Analysing imported (rather than simulated) C coefficients without supplying an `I_scale` parameter
raises:

```
AttributeError: 'density_extraction' object has no attribute 'fig_I0'
  modules/density_extraction.py:2160  in calculate_I0
```

## Cause

`calculate_I0` (`:2123`), in the branch taken when `self.data_or_sim` is true (i.e. real data) and
`"I_scale"` is absent:

```python
#TODO FIX THIS OPTION, fig_I0 does not exist
self.fig_I0(c_calc)
```

The intended method is `fit_I0` (`:2173`). The author knew — the TODO is right there — but it was
never fixed, presumably because every published run supplied `I_scale` explicitly.

This is the single most consequential outstanding bug: BIGR's purpose is retrieving structure from
**measured** diffraction, and this is on that exact path.

## Fix

```python
self.fit_I0(c_calc)
```

`fit_I0(self, c_calc, data=None, var=None, return_vals=False)` defaults `data`/`var` to the
instance's own `data_coeffs`/`data_coeffs_var`, so the one-argument call is consistent with the
signature. Worth confirming the fitted `self.I` is actually assigned on that path (the
`return_vals=False` branch) before trusting it.

## Two sibling instances of the same mistake

`fit_I0` is also called **without `self.`** in two other places, both `NameError` on first use:

| Line | Method | Reached when |
|---|---|---|
| `:2239`, `:2242` | `remove_global_offset` | `global_offset` is in the parameter dict |
| `:2692`, `:2695` | `simulate_error_data` | `simulate_error = ("data", …)` |

Fix all four occurrences together. Note this means the **`("data", (order, Wn))` error model is also
entirely broken** — it is advertised in the parameter list but has never run. `simulate_error_data`
additionally references `plot_folder`, which is a local of `simulate_data` (`:1010`) and not in
scope in that method.

## Test

There is no test suite, so add a minimal round-trip guarding this path: write a small
measured-data HDF5 in the format documented in `how_to_run.md` §10, run with
`simulate_data = False` and no `I_scale`, and assert construction succeeds and `self.I` is finite.

## Related

- `how_to_run.md` §10 — the measured-data HDF5 layout, including the advice to set `I_scale` as a
  workaround
- [010](010-dead-and-broken-code-paths.md) — `remove_global_offset` and `simulate_error_data` have
  further problems beyond this
