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
| [001](001-ston-signal-to-noise-unusable.md) | P1 | error model | `StoN` error model cannot reach usable signal-to-noise with the in-repo ADMs |
| [002](002-L4-coefficients-anomalously-small.md) | **P2** | physics | L=4 C coefficients are ~10⁻⁶ while L=2 and L=6 are ~10⁻² — unexplained |
| [003](003-2dof-symmetric-path-unwired.md) | P1 | geometry | Symmetric (2dof) NO₂ path is unwired — `theta_to_cartesian_2dof` is dead code |
| [004](004-calc-type-1-and-2-broken.md) | P1 | backends | `calc_type` 1 and 2 both crash; `calc_type` is silently ignored when multiprocessing is on |
| [005](005-multiprocessing-broken-on-spawn.md) | P1 | performance | `multiprocessing > 1` cannot work on macOS/Windows (spawn cannot pickle a closure) |
| [006](006-measured-data-path-broken.md) | P1 | real data | The measured-data path crashes: `self.fig_I0` does not exist |
| [007](007-simulated-data-cache-key-incomplete.md) | **P2** | caching | Simulated-data cache silently reuses stale coefficients |
| [008](008-mode-search-no-iteration-cap.md) | P2 | mode search | `weight_avg_search` has no iteration cap or timeout |
| [009](009-ensemble-grid-size-parameter-ignored.md) | P3 | parameters | The ensemble grid size appended to `sim_thetas` never takes effect |
| [010](010-dead-and-broken-code-paths.md) | P3 | dead code | Six dead or broken code paths that raise on first use |
| [011](011-notebook-remaining-problems.md) | P2 | notebook | `analyze_results.ipynb`: stale duplicate functions, undefined names, cluster paths |
| [012](012-setup-sh-broken.md) | P1 | setup | `setup.sh` cannot run: syntax error, undefined variable, bad URLs |
| [013](013-get-adms-molecule-hardcoded.md) | P2 | ADMs | `get_ADMs` hardcodes `"NO2"` and silently drops missing LMK |
| [014](014-scripts-not-importable.md) | P3 | scripts | Driver scripts cannot be imported (module-scope `parse_args`, globals used in `main`) |
| [015](015-scipy-sph-harm-pin.md) | P3 | dependencies | Pinned to scipy < 1.17 by `sph_harm`; also blocks Python > 3.10 |

## Suggested order of work

1. **[006](006-measured-data-path-broken.md)** — one-character fix, and it is the difference
   between the package working on real data or not.
2. **[002](002-L4-coefficients-anomalously-small.md)** — needs your physics judgement, and if it
   *is* a bug it invalidates the L=4 contribution to every published fit.
3. **[012](012-setup-sh-broken.md)**, **[004](004-calc-type-1-and-2-broken.md)** — cheap, and they
   are the first things a new user hits.
4. **[001](001-ston-signal-to-noise-unusable.md)**, **[003](003-2dof-symmetric-path-unwired.md)** —
   feature-level gaps needing real work.
5. The rest as convenient.
