# 014 — Driver scripts cannot be imported

**Severity** P3 (blocks reuse and testing, not the documented workflow)
**Area** scripts
**Status** open

`NO2/build_posterior.py` and `NO2/mode_search.py` work when run as scripts but cannot be imported —
from a notebook, a test, or each other. That matters because `build_posterior.main()` takes a
`return_extraction=True` argument that exists **only** to be used programmatically, so the intent was
clearly there.

## 14a. `argparse.parse_args()` runs at module scope

`NO2/build_posterior.py:20-24`:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--do_ensemble", type=int, default=1, required=False)
parser.add_argument("--do_2dof", type=int, default=0, required=False)
parser.add_argument("--multiProc_ind", type=int, default=None, required=False)
args = parser.parse_args()
```

Importing the module parses the **host's** `sys.argv`. From a Jupyter kernel that is something like
`['.../ipykernel_launcher.py', '-f', '/path/kernel.json']`, so argparse errors on `-f` and calls
`sys.exit(2)` — killing the kernel. Under pytest it consumes pytest's own flags.

`NO2/mode_search.py:15-19` has the same structure.

**Fix.** Move parsing into the entry point and thread the values through:

```python
def main(data_parameters, do_2dof=False, return_extraction=False):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ...
    args = parser.parse_args()
    main(get_parameters(), do_2dof=bool(args.do_2dof))
```

`main()` currently reads `args.do_2dof` directly at `build_posterior.py:52` and `:62`, so those two
lines change with it. As a stopgap, `parser.parse_known_args()` at least stops the hard exit.

## 14b. `main()` reads module globals assigned inside `if __name__ == "__main__"`

`NO2/mode_search.py` reads `use_2dof` at `:47` and `:57`, inside `main()`, but assigns it at `:126`
— inside the `__main__` block. Run as a script it resolves as a module global; **imported, `main()`
raises `NameError`.**

**Fix.** Make it a parameter of `main()`, exactly as in 14a.

## 14c. Importing either module requires a `parameters.py` on `sys.path`

Both do `from parameters import *`, and `modules/mode_search.py:8` does the same. So
`modules.mode_search` — a *library* module — cannot be imported unless some experiment's
`parameters.py` is importable, which in practice means the working directory must be `NO2/`.

This couples the shared library to a per-experiment file. `modules/mode_search.py` uses only a
handful of names from it; taking them from the `data_params` dict it is already handed would remove
the coupling entirely.

## 14d. Fragile star-import re-exports (partly fixed)

`NO2/mode_search.py` used `argparse` and `time` without importing them; they resolved only because
`from modules.NO2 import *` re-exported `modules/NO2.py`'s own imports (that module defines no
`__all__`). Adding an `__all__` there, or reordering, would have broken the script.

Explicit `import argparse, time` were added during this work (see `CHANGELOG.md`). The underlying
fragility remains: several modules rely on `import *` for transitive names. Adding `__all__` to
`modules/NO2.py` and `modules/plot_functions.py` would make the real dependencies visible — but do it
together with an audit, since it will surface more cases like this one.

## Why this is worth fixing

With 14a and 14b fixed, a regression test becomes possible without a cluster:

```python
def test_posterior_recovers_truth(tmp_path):
    p = get_parameters()
    p["output_dir"] = str(tmp_path)
    p["max_iterations"] = 200
    ex = build_posterior.main(p, return_extraction=True)
    truth = np.array(p["sim_thetas"])[:len(p["init_thetas"])]
    assert ex.log_likelihood(truth[None, :])[0] == pytest.approx(0.0, abs=1e-9)
```

That single assertion — `log_likelihood(truth) == 0` for the PDF model — is the check that
established the retrieval was correct during this work, and it runs in seconds. There is currently no
test suite, linter or CI in the repo; this would be a good first test.
