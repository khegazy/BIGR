# 008 — `weight_avg_search` has no iteration cap or timeout

**Severity** P2 (can run indefinitely; no way to bound a cluster job)
**Area** mode search
**Status** open — worked around by killing the process (progress is checkpointed)

## Symptom

`NO2/mode_search.py` can run indefinitely. In the fast configuration Θ\* had stabilised to ~10⁻³
by iteration 5 and was unchanged at iteration 8, but the loop kept going and had to be killed
manually.

## Cause

The main loop in `modules/mode_search.py:265`:

```python
while convergence_count < 3 or np.all(prev_ths_mean == ths_mean):
```

Exit requires `mode_tolerance` to be met on **three consecutive** iterations. There is no iteration
cap, no wall-clock limit, and no minimum-progress test. Two ways it can fail to terminate:

1. Θ oscillating just above tolerance never accumulates three consecutive passes.
2. If `ths_std` collapses toward zero the sampling grid degenerates, so successive iterations
   propose nearly identical Θ. The `np.all(prev_ths_mean == ths_mean)` clause is presumably meant to
   catch that, but it is an **exact float equality** test — it fires only on bit-identical means and
   misses "changed by 1e-15". There is a re-randomisation escape at `:311-320` (after
   `switch_rnd > 5`), which then draws a hardcoded 1500 samples at `:333`.

`density_extraction.run_mcmc` already got a `max_iterations` cap during this work
(`how_to_run.md` §6); the mode search never did.

## Suggested fix

Mirror the MCMC cap. In `weight_avg_search`, add a loop counter and a parameter:

```python
loop_count = 0
max_loops = data_params.get("mode_max_iterations", 50)
while convergence_count < 3 or np.all(prev_ths_mean == ths_mean):
    ...
    loop_count += 1
    if loop_count >= max_loops:
        print("INFO: mode search reached mode_max_iterations ({}), stopping. "
              "convergence_count = {}".format(max_loops, convergence_count))
        break
```

and add `"mode_max_iterations": 50` to `NO2/parameters.py`. Also replace the exact-equality
stall test with a tolerance-based one:

```python
stalled = np.allclose(prev_ths_mean, ths_mean, rtol=0, atol=1e-12)
```

Because `save_mode_search` writes the full state (including `ths_mean_history`) every iteration,
breaking out early is safe and the result is usable — which is exactly what was relied on here.

## Cost knobs worth documenting alongside

| Knob | Where | Default | Effect |
|---|---|---|---|
| `mode_std_grid` | `parameters.py` | `[-1, 0, 1]` | grid is `len(grid) ** ndim` θ per iteration: 3⁶ = **729**; a 5-point grid is 5⁶ = 15 625 |
| `mode_tolerance` | `parameters.py` | 1e-4 (0.01 here) | must be met 3 consecutive times |
| `N_mode_samples` | `parameters.py` | 50 (25 here) | posterior samples seeding each iteration |
| `stride` | `modules/mode_search.py:41` | 200 | θ per likelihood batch; lower it if memory-bound |
| random fallback | `modules/mode_search.py:333` | 1500 | hardcoded, not a parameter |

The last two are hardcoded and should be parameters.

## Related

- `how_to_run.md` §7 ("Mode search cost") documents the grid scaling for users
