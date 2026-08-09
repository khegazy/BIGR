# Known issues

Issues found while getting the NO₂ analysis running off the original SLAC/LCLS cluster (see
`how_to_run.md` for the procedure and for the bugs that were *already fixed*). Everything here is
**still outstanding**.

Each file is self-contained: what breaks, where, why, how to reproduce, and a suggested fix.
Line numbers refer to the `ressurect` branch as of the commit that added this folder.

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
| [007](007-simulated-data-cache-key-incomplete.md) | **P2** | caching | Simulated-data cache silently reuses stale coefficients |
| [008](008-mode-search-no-iteration-cap.md) | ~~P2~~ | mode search | ✅ **FIXED** — `mode_max_iterations` cap; Θ* now reported against the best sample |
| [009](009-ensemble-grid-size-parameter-ignored.md) | P3 | parameters | The ensemble grid size appended to `sim_thetas` never takes effect |
| [010](010-dead-and-broken-code-paths.md) | P3 | dead code | ⚠️ **partly fixed** — 10a, 10d and 10f resolved; 10b, 10c, 10e, 10g remain |
| [011](011-notebook-remaining-problems.md) | P2 | notebook | `analyze_results.ipynb`: stale duplicate functions, undefined names, cluster paths |
| [012](012-setup-sh-broken.md) | ~~P1~~ | setup | ✅ **FIXED** — rewritten; parses, runs, idempotent |
| [013](013-get-adms-molecule-hardcoded.md) | ~~P2~~ | ADMs | ✅ **13a/13b FIXED** — raises on a missing LMK; molecule dir configurable |
| [014](014-scripts-not-importable.md) | P3 | scripts | Driver scripts cannot be imported (module-scope `parse_args`, globals used in `main`) |
| [015](015-scipy-sph-harm-pin.md) | P3 | dependencies | Pinned to scipy < 1.17 by `sph_harm`; also blocks Python > 3.10 |
| [016](016-ensemble-quadrature-error-dominates-likelihood.md) | ~~P1~~ | likelihood | ❌ **RETRACTED** — was an artifact of the centre-of-mass bug |
| [017](017-mode-search-returns-worse-than-median.md) | P2 | mode search | `weight_avg_search` returns a centroid, not a maximum — **numbers need re-verification** |
| [018](018-posterior-prior-dominated-at-low-information.md) | ~~P2~~ P3 | priors | ⚠️ **revised** — measurement invalid; the guard is still worth adding |
| [019](019-prefactor-convention-differs-from-eq21.md) | P2 | conventions | C-coefficient prefactor differs from Eq. 21 by an L-dependent normalisation |

## Suggested order of work

**Update 2026-08-08.** A single root-cause fix — a centre-of-mass bug in `rotate_to_principalI`
that scaled every molecule by 1/total_mass — resolved or retracted five of these
([002](002-L4-coefficients-anomalously-small.md) resolved,
[016](016-ensemble-quadrature-error-dominates-likelihood.md) retracted,
[001](001-ston-signal-to-noise-unusable.md) largely resolved,
[018](018-posterior-prior-dominated-at-low-information.md) revised,
[017](017-mode-search-returns-worse-than-median.md) needs re-verification). Several others were
fixed directly. What remains:

1. **[019](019-prefactor-convention-differs-from-eq21.md)** — needs an authors' decision on which
   spherical-harmonic convention Eq. 21 assumes. Matters for imported measured data.
2. **[003](003-2dof-symmetric-path-unwired.md)** — the symmetric 2dof path is still unwired.
3. **[017](017-mode-search-returns-worse-than-median.md)** — re-measure on a valid posterior, then
   add the guard.
4. **[005](005-multiprocessing-broken-on-spawn.md)**, **[007](007-simulated-data-cache-key-incomplete.md)**,
   **[008](008-mode-search-no-iteration-cap.md)**, **[013](013-get-adms-molecule-hardcoded.md)** —
   real but independent of the above.
5. **[004b](004-calc-type-1-and-2-broken.md)**, **[010b/e/g](010-dead-and-broken-code-paths.md)** —
   dead code: the recommendation is to delete rather than repair.
