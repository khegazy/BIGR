# 003 — Symmetric (2dof) NO₂ path is unwired; `theta_to_cartesian_2dof` is dead code

**Severity** P1 (documented parameter value cannot be used)
**Area** geometry / degrees of freedom
**Status** open

## Symptom

Setting `experiment = "2dof"` in `NO2/parameters.py` — a value the file itself handles at
`:134-144`, and which `build_posterior.py` and `mode_search.py` both branch on — raises:

```
IndexError: index 3 is out of bounds for axis 1 with size 3
  modules/NO2.py:366  in molecule_ensemble_generator
    d2, std_2 = thetas[:,2], thetas[:,3]
```

## Cause

The 2dof parameterisation shares a single bond length between both oxygens. There *is* a Cartesian
builder for it — `theta_to_cartesian_2dof` (`modules/NO2.py:196`), which reuses `theta[:,0]` for
both O atoms at lines 213-216 — but **no generator ever calls it**:

```
$ grep -n "theta_to_cartesian" modules/NO2.py
146:def theta_to_cartesian_single(theta)      # 3dof: independent theta[:,0], theta[:,1]
171:def theta_to_cartesian_ensemble(theta)    # 3dof
196:def theta_to_cartesian_2dof(theta)        # 2dof  <-- never called
342:  molecules = theta_to_cartesian_single(thetas)
411:  molecules = theta_to_cartesian_ensemble(new_thetas)
```

So `experiment="2dof"` selects the 2dof **log priors** (`log_prior_2dof_gauss` /
`log_prior_2dof_delta`, wired at `NO2/build_posterior.py:52-63`) and a 4-element `init_thetas`,
but the **generators still build 3dof geometries** and index `thetas[:,3]`/`thetas[:,5]`.

This matters because the two NO₂ geometries in the repo genuinely differ:

| | `NO2/XYZ/NO2.xyz` | `NO2/XYZ/NO2_symbreak.xyz` |
|---|---|---|
| N–O | both **1.1934 Å** | **1.35** and **1.05 Å** |
| ∠ONO | 2.337 rad | 2.337 rad |
| correct `experiment` | `"2dof"` | `"3dof"` |
| ndim (PDF / delta) | 4 / 2 | 6 / 3 |

A symmetric molecule fitted with 3dof wastes a degree of freedom and introduces an artificial
d₁ ↔ d₂ degeneracy; an asymmetric one fitted with 2dof cannot represent the structure at all.

## Suggested fix

Add the two missing generators next to the existing ones in `modules/NO2.py`, mirroring
`single_molecule_generator` (`:342`) and `molecule_ensemble_generator` (`:411`) but with a
2-parameter grid and `theta_to_cartesian_2dof`:

```python
def single_molecule_generator_2dof(thetas):
    molecules = theta_to_cartesian_2dof(thetas)
    return np.expand_dims(molecules, 1), np.ones((thetas.shape[0], 1))


def molecule_ensemble_generator_2dof(thetas):
    # thetas = [d, sigma_d, angle, sigma_angle]; outer product over 2 dof -> N**2 geometries
    d,   std_d = thetas[:, 0], thetas[:, 1]
    ang, std_a = thetas[:, 2], thetas[:, 3]
    N, std_ = ENSEMBLE_GRID_N, 7
    ...
    molecules = theta_to_cartesian_2dof(new_thetas)
    return molecules, joint_probs
```

then select them in `NO2/build_posterior.py:44-68` and `NO2/mode_search.py:41-67` alongside the
already-correct `log_prior_2dof_*` choice.

Bonus: the 2dof ensemble is only N² rather than N³, so it is ~N× cheaper per likelihood
evaluation — a useful fast configuration in its own right (see
[009](009-ensemble-grid-size-parameter-ignored.md)).

## Also worth checking while in there

`log_prior_2dof_delta` (`modules/NO2.py:118`) rejects on `theta[:,1] > pi` or `< 1.0`. For the
2-element delta vector `(d, angle)`, index 1 is the angle, so the `> pi` bound is right — but the
`< 1.0` lower bound on an **angle in radians** (57°) is much tighter than the 3dof version's, and
looks like it may have been copied from a distance bound. Confirm it is intentional.

## Related

- `how_to_run.md` §4 documents the distinction for users
- [009](009-ensemble-grid-size-parameter-ignored.md) — ensemble cost scaling
