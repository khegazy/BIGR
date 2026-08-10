# 015 — Effectively pinned to scipy < 1.17 and Python 3.10 by `sph_harm`

**Severity** P3 (works today; a scheduled breakage)
**Area** dependencies
**Status** partly addressed — the repo's own call sites are migrated; the constraint now comes from
`external_artifacts/` and from Python version support

## Current state

`scipy.special.sph_harm` was deprecated in scipy 1.15 and **removed in 1.17**. During this work all
call sites were migrated to `sph_harm_y`:

- `modules/density_extraction.py:1434, 1489, 1622` — migrated
- `external_artifacts/modules/diffraction_simulation.py:37, 419, 427` — migrated

The migration was verified rather than assumed, because `sph_harm_y` swaps **both** the degree/order
and the angle order:

```
sph_harm(m, n, azim, polar)  ==  sph_harm_y(n, m, polar, azim)
```

Agreement was 1.1e-15 over l = 0…6, confirmed against the analytic Y₁⁰/Y₁¹/Y₂⁰ closed forms and
re-checked under each call site's real broadcast shapes (5.6e-16). The naive *unswapped* call differs
by 0.243, so this was a genuine trap. Details in `CHANGELOG.md`.

## What still constrains the versions

1. **Python 3.10.** The venv is 3.10.20, and **scipy 1.15.3 is the last release supporting 3.10**.
   So the environment cannot move to newer scipy without first moving Python. Nothing in the code
   requires 3.10; it is simply what the venv was created with.
2. **`external_artifacts/modules/` is vendored, pinned code.** Both files were recovered from 2022-23
   archives and modernised in place. They are not tracked upstream any more, so future numpy/scipy
   removals will have to be fixed here by hand. `diffraction_simulation.py` in particular still uses
   `np.complex64`/`np.int16` dtypes (valid, but the module is clearly of that vintage) and
   `sp.special.factorial`, `sp.special.jacobi`.
3. **No lower bounds are recorded anywhere.** `README.md` lists floors (numpy ≥ 1.21.4,
   scipy ≥ 1.6.2) that are now wrong in both directions: the code no longer runs on numpy < 1.24
   idioms it used to depend on, and it will not run on scipy ≥ 1.17 if the migration is ever
   reverted. There is no `requirements.txt`, `environment.yml` or `pyproject.toml`.

## Suggested fix

1. **Record the environment.** Add a `requirements.txt` with the versions actually verified:
   ```
   numpy==2.2.6
   scipy==1.15.3
   matplotlib==3.10.9
   h5py==3.16.0
   emcee==3.1.6
   corner==2.3.0
   tqdm
   ```
   and note in `README.md` that these are the tested versions, superseding its old floors.
2. **Re-verify on upgrade.** When moving to scipy ≥ 1.17, re-run the equivalence check before
   trusting anything — `sph_harm` will be gone, so compare against the analytic closed forms:
   ```python
   from scipy.special import sph_harm_y
   import numpy as np
   th, ph = 0.7, 1.3
   assert np.isclose(sph_harm_y(1, 0, th, ph),  np.sqrt(3/(4*np.pi))*np.cos(th))
   assert np.isclose(sph_harm_y(1, 1, th, ph), -np.sqrt(3/(8*np.pi))*np.sin(th)*np.exp(1j*ph))
   assert np.isclose(sph_harm_y(2, 0, th, ph),  np.sqrt(5/(16*np.pi))*(3*np.cos(th)**2 - 1))
   ```
   Keeping this as a unit test would make the constraint self-checking — see
   [014](014-scripts-not-importable.md) for why there is no test suite to put it in yet.
3. **Decide about `external_artifacts/`.** Either accept it as vendored and maintained here (then
   add its provenance commit hashes to the README so upstream diffs are traceable — currently only
   the archive zip paths are recorded), or extract the two functions actually used
   (`diffraction_calculation`, `fit_legendres_images`) into `modules/` as first-class BIGR code.
   Only ~3 of the 452 lines in `diffraction_simulation.py` are on the live path.

## Note on the C++ extension

Unrelated to scipy but in the same category: `cpp_extensions/lib/c_calc_extensions.{so,o}` were
untracked during this work because they had been committed as Linux x86-64 binaries. Each platform
now builds its own. The `Makefile` also contains a harmless typo, `$(patsusbst ...)` at line 11,
which GNU make expands to empty; `T_FILES` is only used in an `$(info)` so nothing breaks.
