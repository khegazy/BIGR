# Known issues

Issues found while getting the NO₂ analysis running off the original SLAC/LCLS cluster.
**Many have since been fixed** — the Status line at the top of each file is authoritative, and the
✅/⚠️/❌ markers in the index below summarise it. `CHANGELOG.md` records the code changes themselves.

Each file is self-contained: what breaks, where, why, how to reproduce, and a suggested fix. Fixed
issues are kept rather than deleted, because several of the original diagnoses turned out to be
wrong and the retraction is worth reading — see 016 and 017 in particular.

Line numbers drift as fixes land. Where a fix moved code, the Status line gives the current line;
elsewhere numbers refer to the branch as of the commit that added this folder.

Last audited against the code: **2026-08-09**.

## Severity

- **P1 — blocks a documented feature.** The feature is advertised in `README.md` or the
  parameter list but cannot work at all.
- **P2 — correctness risk.** Runs, but can silently produce wrong numbers.
- **P3 — papercut.** Confusing, fragile, or dead code; no wrong results.

## Index

| # | Severity | Area | Title |
|---|---|---|---|
| [001](001-ston-signal-to-noise-unusable.md) | ~~P1~~ | error model | ✅ **largely resolved** — StoN works; the low S/N was the centre-of-mass bug |
| [002](002-L4-coefficients-anomalously-small.md) | ~~P1~~ | physics | ✅ **RESOLVED** — centre-of-mass bug scaled every molecule by 1/total_mass |
| [003](003-2dof-symmetric-path-unwired.md) | P1 | geometry | Symmetric (2dof) NO₂ path is unwired — `theta_to_cartesian_2dof` is dead code |
| [004](004-calc-type-1-and-2-broken.md) | ~~P1~~ P3 | backends | ⚠️ **4a/4c fixed** — `calc_type=1` works; 2 recommended for deletion |
| [005](005-multiprocessing-broken-on-spawn.md) | P1 | performance | `multiprocessing > 1` cannot work on macOS/Windows (spawn cannot pickle a closure) |
| [006](006-measured-data-path-broken.md) | ~~P1~~ | real data | ✅ **FIXED** — plus a second bug behind it in `fit_I0` |
| [007](007-simulated-data-cache-key-incomplete.md) | ~~P2~~ | caching | ✅ **FIXED** — content hash; a parameter change is now a cache miss |
| [008](008-mode-search-no-iteration-cap.md) | ~~P2~~ | mode search | ✅ **FIXED** — `mode_max_iterations` cap; Θ* now reported against the best sample |
| [009](009-ensemble-grid-size-parameter-ignored.md) | P3 | parameters | The ensemble grid size appended to `sim_thetas` never takes effect |
| [010](010-dead-and-broken-code-paths.md) | P3 | dead code | ⚠️ **partly fixed** — 10a, 10c, 10d, 10f resolved; 10b **half** (still raises `TypeError`); 10e, 10g remain |
| [011](011-notebook-remaining-problems.md) | P2 | notebook | `analyze_results.ipynb`: stale duplicate functions, undefined names, cluster paths |
| [012](012-setup-sh-broken.md) | ~~P1~~ | setup | ✅ **FIXED** — rewritten; parses, runs, idempotent |
| [013](013-get-adms-molecule-hardcoded.md) | ~~P2~~ | ADMs | ✅ **13a/13b FIXED** — raises on a missing LMK; molecule dir configurable |
| [014](014-scripts-not-importable.md) | P3 | scripts | Driver scripts cannot be imported (module-scope `parse_args`, globals used in `main`) |
| [015](015-scipy-sph-harm-pin.md) | P3 | dependencies | ⚠️ **partly addressed** — all `sph_harm` call sites migrated; the pin now stands only on Python 3.10 |
| [016](016-ensemble-quadrature-error-dominates-likelihood.md) | ~~P1~~ | likelihood | ❌ **RETRACTED** — was an artifact of the centre-of-mass bug |
| [017](017-mode-search-returns-worse-than-median.md) | ~~P2~~ P3 | mode search | ❌ **not reproducible** — Θ\* now beats the median by ~5×10⁶; only the structural caveat stands |
| [018](018-posterior-prior-dominated-at-low-information.md) | ~~P2~~ P3 | priors | ⚠️ **revised** — measurement invalid; the guard is still worth adding |
| [019](019-prefactor-convention-differs-from-eq21.md) | P2 | conventions | C-coefficient prefactor differs from Eq. 21 by an L-dependent normalisation |
| [020](020-per-lmk-plots-all-identical.md) | ~~P2~~ | plotting | ✅ **fixed** — every per-LMK diagnostic plot showed the same (2,0,0) curve |

## Suggested order of work

**Update 2026-08-09.** A single root-cause fix — a centre-of-mass bug in `rotate_to_principalI` that
scaled every molecule by 1/total_mass — resolved or retracted five of these
([002](002-L4-coefficients-anomalously-small.md) resolved,
[016](016-ensemble-quadrature-error-dominates-likelihood.md) retracted,
[001](001-ston-signal-to-noise-unusable.md) largely resolved,
[018](018-posterior-prior-dominated-at-low-information.md) revised,
[017](017-mode-search-returns-worse-than-median.md) does not reproduce). Several others were fixed
directly: 006, 007, 008, 012, 013a/b, 020 and four of the six paths in 010. A converged 3000-step run
then recovered all six parameters within 1σ of truth, and the mode search found Θ\* to 5–6
significant figures.

**Fixed and verified:** 001, 002, 004a, 004c, 006, 007, 008, 010a, 010c, 010d, 010f, 012, 013a,
013b, 020. **Retracted as wrong:** 016, 017.

What actually remains, in the order worth doing it:

1. **[019](019-prefactor-convention-differs-from-eq21.md)** (P2) — needs an authors' decision on
   which spherical-harmonic convention Eq. 21 assumes. **The only open issue that can silently
   change published numbers**, and it bites hardest on imported measured data.
2. **[003](003-2dof-symmetric-path-unwired.md)** (P1) — the symmetric 2dof path still raises
   `IndexError`; `theta_to_cartesian_2dof` has no caller. A documented parameter value that cannot
   run.
3. **[011](011-notebook-remaining-problems.md)** (P2) — two notebook cells still shadow
   `plot_functions` with swapped arguments, which draws a **wrong figure with no error**. Cheap to
   fix by deleting the stale cells.
4. **Cheap guards that turn confusing failures into clear ones** — none of these is a wrong-number
   risk, but each currently fails in a way that wastes an afternoon:
   - [005](005-multiprocessing-broken-on-spawn.md) (P1) — raise a clear error when
     `multiprocessing > 1` on a spawn platform, instead of an opaque `PicklingError`.
   - [010b](010-dead-and-broken-code-paths.md) — drop the `N=` keyword the generator does not
     accept, and delete the leftover `print("AAAA", …)` beside it. Also closes
     [009](009-ensemble-grid-size-parameter-ignored.md).
   - [017](017-mode-search-returns-worse-than-median.md),
     [018](018-posterior-prior-dominated-at-low-information.md) — the result guards; the reported
     failures no longer reproduce, but both would fail silently if they returned.
5. **[014](014-scripts-not-importable.md)** (P3) — `parse_args()` at module scope means importing a
   driver script parses the *host's* argv and kills a Jupyter kernel with `SystemExit(2)`. Verified
   still broken 2026-08-09.
6. **[013c](013-get-adms-molecule-hardcoded.md)**, **[015](015-scipy-sph-harm-pin.md)** (P3) —
   latent: ADM index parsing breaks at L or K ≥ 10; the scipy pin now rests only on Python 3.10.
7. **[004b](004-calc-type-1-and-2-broken.md)**, **[010e](010-dead-and-broken-code-paths.md)**,
   **[010g](010-dead-and-broken-code-paths.md)** — dead code. The recommendation is to **delete
   rather than repair**: `calc_type = 2` still fails with an `IndexError` after three fixes,
   `save_results`/`get_results` have no callers, and 010g is a block behind `and False`.
