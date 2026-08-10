# 013 — `get_ADMs` hardcodes `"NO2"` in its path and silently drops missing LMK

**Severity** ~~P2~~
**Area** ADMs
**Status** ✅ **FIXED 2026-08-08** for 13a and 13b. The molecule sub-directory is now
`params.get("molecule_dir", "NO2")` — backwards compatible, but no longer a literal — with a comment
recording that the ADMs describe *orientation* and so are shared between the symmetric and stretched
NO₂ geometries. And a requested (L,K) that does not match exactly one ADM file now raises with the
available (L,K) listed, instead of appending an empty slice and silently returning fewer rows than
`get_LMK`. Verified: a normal load returns matching row counts, and requesting an absent (L,K) raises.

13c (positional `int(fl[-6])`/`int(fl[-5])` index parsing, which breaks for L or K ≥ 10) is still
open.

## 13a. `"NO2"` is hardcoded in the path

`modules/NO2.py:555`:

```python
folders = [params["folder"], "NO2", "ADMs",
    "temp-{}K".format(temp_str),
    "{}TW_{}fs".format(int(params["intensity"]), fwhm_str)]
```

`"NO2"` is a literal, not `params["molecule"]`. Two consequences:

1. Running with `molecule = "NO2_symbreak"` (the default!) still reads ADMs from a directory called
   `NO2/`. That happens to be correct — the ADMs describe *orientation* and are computed from the
   real molecule's rotational constants, so they are shared between the symmetric and stretched
   geometries — but nothing says so, and it reads like a bug.
2. Anyone following the per-experiment pattern in `README.md` for a different molecule inherits a
   function that silently looks in `NO2/`. Since `modules/NO2.py` is explicitly the template for new
   experiments (`modules/module_template.py` declares the same `get_ADMs` contract), this will bite.

**Fix.** Either take the subdirectory from the parameters:

```python
folders = [params["folder"], params.get("molecule_dir", "NO2"), "ADMs", ...]
```

or, better, drop the molecule level entirely and let `ADM_params["folder"]` point at the directory
that directly contains `temp-*K/`. The extra `NO2/ADMs/` nesting buys nothing and is the reason
`scripts/stage_adms.py` has to build a two-level tree.

Add a comment stating explicitly that the ADMs are geometry-independent, so the next reader does not
"fix" it wrongly.

## 13b. A missing LMK silently shortens the returned array

`modules/NO2.py:617-625`:

```python
for lmk_ in get_LMK:
    lInds = LMK[:,0] == lmk_[0]
    mInds = LMK[:,2] == lmk_[2]
    fit_bases.append(allBases[lInds*mInds])
    fit_norms.append(allNorms[lInds*mInds])
LMK = get_LMK
return LMK, np.concatenate(fit_bases, axis=0), np.concatenate(fit_norms, axis=0), params["eval_times"]
```

If a requested `(L, K)` has no corresponding file, `allBases[lInds*mInds]` is an **empty slice**.
`np.concatenate` accepts it, so the returned `bases` has **fewer rows than `get_LMK`** — while the
returned `LMK` is `get_LMK` at full length. The two are now misaligned.

Downstream, `simulate_error_StoN` uses `self.ADMs` both as diffraction weights (`:2857-2862`) and as
the design matrix `A` in `inv(AᵀWA)` (`:2946-2953`), indexing it by `self.data_LMK`. A silent
off-by-one in that pairing associates each ADM with the **wrong** LMK, which corrupts the error
propagation rather than raising.

This did not bite here only because all 14 `A*.npy` files happen to be present at every
temperature, covering the six `fit_bases` triplets.

Note also that the match is on **L and K only** — `mInds` compares `LMK[:,2]`, i.e. K, despite the
variable name suggesting M. M is always 0 in this dataset, so it is currently harmless, but the
naming is actively misleading.

**Fix.** Fail loudly:

```python
for lmk_ in get_LMK:
    sel = (LMK[:, 0] == lmk_[0]) & (LMK[:, 2] == lmk_[2])
    if sel.sum() != 1:
        raise ValueError(
            "get_ADMs: requested LMK {} matched {} ADM files in {}; expected exactly 1".format(
                lmk_, sel.sum(), folderName))
    fit_bases.append(allBases[sel])
    fit_norms.append(allNorms[sel])
```

`prune_data:1250` already does exactly this kind of check for `fit_bases` against `data_LMK`, so the
pattern is established in the codebase.

## 13c. Minor: positional index parsing is brittle

`:577-578`:

```python
L = int(fl[-6])
K = int(fl[-5])
```

L and K are read as **single characters** at fixed offsets from the end of the path. This works for
`… D62.npy` → L=6, K=2, but breaks silently for any L or K ≥ 10, and depends on the filename ending
in exactly two digits plus `.npy`. `scripts/stage_adms.py` therefore has to preserve the original
filenames verbatim, which is documented but fragile.

A regex would be clearer and would fail loudly on an unexpected name:

```python
m = re.search(r"D(\d+)(\d)\.npy$", fl)     # or a documented explicit separator
```

Better still, record L and K inside the `.npy` files (or a small sidecar JSON) at generation time
rather than encoding them in filenames.

## Related

- `scripts/stage_adms.py` — documents and works around 13c
- [001](001-ston-signal-to-noise-unusable.md) — the error propagation that 13b would corrupt
