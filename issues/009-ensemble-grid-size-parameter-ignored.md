# 009 — The ensemble grid size appended to `sim_thetas` never takes effect

**Severity** P3 (confusing; the intended knob does nothing)
**Area** parameters / performance
**Status** partly addressed — the value is now a documented constant, but the `sim_thetas`
mechanism is still dead

## Symptom

`NO2/parameters.py:146-147` appends `51` to `sim_thetas`, evidently intending to set the ensemble
grid size:

```python
data_parameters["sim_thetas"] = np.concatenate([data_parameters["sim_thetas"], [51]])
```

`molecule_ensemble_generator` does look for it (`modules/NO2.py:368-372`):

```python
if thetas.shape[-1] == 7:
    N = int(thetas[0, 6])
    thetas = thetas[:, :-1]
else:
    N = ENSEMBLE_GRID_N
```

But the 7-element branch is never reached, so **N was always the hardcoded fallback** (19 before
this work, `ENSEMBLE_GRID_N = 11` now). The `51` is inert.

## Cause

Every consumer strips the trailing element before the generator sees it:

- `simulate_data:1002` passes `sim_thetas[:-1]` → 6 elements.
- `init_thetas` is copied at `parameters.py:130`, **before** the append → 6 elements → MCMC
  `ndim = 6`, so sampled θ are always 6 long.

The only place that would have supplied 7 is `remove_global_offset:2225`, which calls
`self.ensemble_generator(..., N=...)` — but `molecule_ensemble_generator` accepts **no `N`
keyword**, so that path raises `TypeError` anyway (see
[010](010-dead-and-broken-code-paths.md)).

This matters more than it looks: the grid size is the dominant cost driver, scaling as N³ for
3 degrees of freedom. Anyone reading `parameters.py` would reasonably believe they were running a
51³ = 132 651-point ensemble when it was in fact 19³ = 6859.

| N | geometries/walker | s per MCMC step (32 walkers, 58 q points) |
|---|---|---|
| 7 | 343 | ~0.15 |
| 11 | 1331 | 0.56 |
| 19 | 6859 | 2.9 |
| 51 | 132 651 | ~56 (extrapolated) |

## Current state

`modules/NO2.py` now defines `ENSEMBLE_GRID_N = 11` as a documented module constant with the cost
table and a warning that changing it invalidates the simulated-data cache
([007](007-simulated-data-cache-key-incomplete.md)). That makes the effective value discoverable,
but it is still a module constant rather than a runtime parameter, and the misleading `51` is still
appended.

## Suggested fix

Make it a normal runtime parameter and delete the `sim_thetas` trick:

1. Add `"ensemble_grid_N": 11` to the `parameters.py` dict.
2. Remove `parameters.py:146-147`.
3. Pass it explicitly. Since the generator signature is fixed by the `density_extraction`
   callback contract (`density_generator(thetas)`), the clean options are either a
   `functools.partial`:
   ```python
   input_density_generator = partial(molecule_ensemble_generator,
                                     N=data_parameters["ensemble_grid_N"])
   ```
   (which also requires giving the generator a real `N=None` keyword — and would incidentally fix
   `remove_global_offset`), or reading it from the params dict the module already has access to.
4. Then delete the dead `thetas.shape[-1] == 7` branch.

Option 3-via-`partial` is preferable because it makes the value explicit at the call site in
`build_posterior.py`, where the rest of the generator wiring already happens.

## Related

- [007](007-simulated-data-cache-key-incomplete.md) — N is not in the cache key
- [010](010-dead-and-broken-code-paths.md) — the `N=` kwarg call in `remove_global_offset`
