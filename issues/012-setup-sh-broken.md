# 012 — `setup.sh` cannot run: syntax error, undefined variable, bad URLs

**Severity** P1 (it is the documented first step for every new user)
**Area** setup
**Status** open — `how_to_run.md` §1 gives replacement commands; `setup.sh` itself is untouched

`README.md` says: *"**Setup** run the setup script `bash setup.sh`"*. It cannot work. Every problem
below was verified by reading the file (37 lines).

## 12a. Unterminated string on the last line — **fatal syntax error**

Line 37:

```bash
echo "INFO: C++ libraries compiled correctly!!! Please use option 0 for runtime parameter 'calc_type'!!!
```

The closing `"` is missing, so `bash` fails to parse the file. Depending on the shell this either
aborts immediately or swallows the rest of the script.

Fix: add the closing quote.

## 12b. `$FILE` is never defined, so two symlinks silently never happen

Lines 15-17 and 25-27 test `$FILE`:

```bash
if test -f "$FILE"; then
  ln -s $FILE parameters_N2O_data.py
fi
```

but the variable defined at line 1 is `DATA_PARAMS_FILE`. `$FILE` expands to empty, `test -f ""` is
false, and the `ln -s` never runs — **without any message**. This is why
`NO2/parameters_N2O_data.py` does not exist, which in turn is why the notebook's N₂O measured-data
section cannot run ([011d](011-notebook-remaining-problems.md)).

Fix: `if test -f "$DATA_PARAMS_FILE"; then ln -s "$DATA_PARAMS_FILE" parameters_N2O_data.py; fi`

## 12c. Both `wget` URLs are wrong, in two different ways

Lines 31-32:

```bash
wget https://githubi.com/khegazy/UED_analysis/blob/ad77b4b.../modules/fitting.py
wget https://github.com/khegazy/physics_simulations/blob/42a2a0e.../diffraction_simulation.py
```

1. Line 31's host is **`githubi.com`** — a typo for `github.com`.
2. Both are GitHub **`/blob/`** URLs, which serve an HTML page, not the file. Even with the host
   fixed you would download HTML named `fitting.py`, and the subsequent import would fail with a
   syntax error rather than a missing-file error.

Fix: use `raw.githubusercontent.com`:

```bash
wget https://raw.githubusercontent.com/khegazy/UED_analysis/ad77b4ba4cb63a96afb74128605580fb6f881bd1/modules/fitting.py
wget https://raw.githubusercontent.com/khegazy/physics_simulations/42a2a0ef68e18f75f8ab8b3836672fa502ae1164/diffractionSimulation/modules/diffraction_simulation.py
```

Both files are now vendored in `external_artifacts/modules/` (see `external_artifacts/README.md`),
so the downloads can simply be **deleted** instead. Note the vendored copies also carry API
modernisations that the upstream pinned commits do not — re-downloading them would reintroduce the
numpy/scipy incompatibilities.

## 12d. `mkdir output` is not idempotent

Line 8 lacks `-p`, unlike lines 6, 7 and 12, so re-running the script errors with
`mkdir: output: File exists`.

Fix: `mkdir -p output`. Also add `mkdir -p output/logs`, which
`NO2/submitClusterJobs*.sh` writes into (`-o ./output/logs/...`) and which nothing creates.

## 12e. `make` result is never checked

Lines 34-37 print "C++ libraries compiled correctly!!!" unconditionally after `make clean; make`,
even on failure — the opposite of the message on line 34, which promises a graceful fallback.

Fix:

```bash
if make clean && make; then
  echo "INFO: C++ libraries compiled; use calc_type = 0"
else
  echo "WARNING: C++ build failed. See issues/004 -- calc_type 1 and 2 are also broken," \
       "so there is currently no working fallback."
fi
```

(The claimed fallback does not presently exist — see
[004](004-calc-type-1-and-2-broken.md).)

## 12f. Fragile `cd` chain, and a stale directory assumption

The script `cd`s without checking: line 11 `cd NO2`, line 20 `cd plots/analysis`, line 30
`cd ../../../modules`. Line 20 assumes `NO2/plots/analysis` already exists — it happens to be in
the repo, but line 12 only created `NO2/plots`. Line 30 lands in `NO2/modules`, which is itself the
symlink created on line 13, so it resolves to `modules/` only if line 13 succeeded.

Fix: use absolute paths from the script's own location, and `set -euo pipefail`:

```bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

## Also missing relative to what is actually needed

`setup.sh` never rebuilds for the host platform in a way that fails loudly, and the repo shipped a
Linux x86-64 `.so` (now untracked). And it does not create `output/saved_simulations`, which
`save_simulated_data` does create on demand — fine, but worth noting for the cluster job scripts.

## Suggested resolution

Rewrite `setup.sh` to match the verified sequence in `how_to_run.md` §1 (symlinks, directories,
`make`), drop the `wget` block entirely now that the modules are vendored, add `set -euo pipefail`,
and run it in CI so it cannot rot again.
