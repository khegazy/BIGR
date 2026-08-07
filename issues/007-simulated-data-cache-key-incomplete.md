# 007 — Simulated-data cache silently reuses stale coefficients

**Severity** P2 (silently wrong results, no error)
**Area** caching
**Status** open

## Symptom

Change something that alters the simulated C coefficients, re-run, and the analysis loads the
**previous** coefficients from disk without any warning. The MCMC then fits new-model to old-data
and the retrieval is quietly wrong.

`README.md` mentions this for the ensemble generator ("make sure to delete this folder"), but the
list of parameters that trigger it is longer than documented and there is no safeguard.

## Cause

The cache path is `os.path.join(save_sim_data, get_fileName() + ".h5")` (`:1178`), and
`get_fileName` (`:2317`) encodes only:

```
<molecule>/<experiment>/<sim|data>/<density_model>/<T>K_<I>TW_<FWHM>fs/
results_<dtype>_range-<lo>-<hi>_error-<etype>_scale-<scale>
```

So these **are** in the key: `molecule`, `experiment`, `density_model`, ADM `temperature`,
`intensity`, `probe_FWHM`, `fit_range`, error type and error scale.

And these **are not**, despite all changing the coefficients:

| Parameter | Why it changes the coefficients |
|---|---|
| `fit_bases` | which LMK are computed at all |
| `sim_thetas` | the ground-truth geometry the data is simulated from |
| `ADM_params["eval_times"]` | the time window and sampling of the ADM fit |
| `ENSEMBLE_GRID_N` (`modules/NO2.py`) | the discretisation of P^(N) |
| `q_per_pix` | the number of `dom` points |
| `isMS`, `q_scale` | normalisation |
| the `ensemble_generator` callback itself | everything |

A `q_per_pix` change usually surfaces as a shape mismatch in `prune_data` (`:1266`), so it fails
loudly. The others do not — they produce a same-shaped array of wrong numbers.

Related dead giveaway that the naming was meant to be richer: `get_fileName` computes an LMK label
and never uses it (`:2341-2343`):

```python
lg_name = "lg"
for l in ...:
    lg_name += "-{}".format(int(l))
# lg_name is never referenced again
```

## Suggested fix

Store a **content hash** of everything the coefficients depend on, and treat a mismatch as a cache
miss. This is more robust than lengthening the filename, and it cannot go stale as parameters are
added:

```python
def _sim_cache_key(self):
    import hashlib, json
    p = self.data_params
    payload = {
        "fit_bases":   np.asarray(p["fit_bases"]).tolist(),
        "sim_thetas":  np.asarray(p["sim_thetas"]).tolist(),
        "dom":         np.asarray(p["dom"]).tolist(),
        "isMS":        bool(p["isMS"]),
        "q_scale":     float(p["q_scale"]),
        "eval_times":  np.asarray(p["ADM_params"]["eval_times"]).tolist()
                       if "ADM_params" in p else None,
        "generator":   getattr(self.ensemble_generator, "__name__", None),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
```

Write it as an attribute in `save_simulated_data` and compare in `load_simulated_data`:

```python
# save
h5.attrs["sim_key"] = self._sim_cache_key()

# load
if h5.attrs.get("sim_key") != self._sim_cache_key():
    print("INFO: simulated-data cache is stale (parameters changed), re-simulating")
    return False
```

Even without the hash, a one-line improvement is to **print what was loaded** rather than silently
substituting it — `load_simulated_data` currently logs the filename but not the fact that this
bypasses simulation entirely.

## Workaround in place

Documented in `how_to_run.md` §11, and the relevant `parameters.py` comment warns about
`eval_times`/`probe_FWHM` specifically. The manual remedy is:

```bash
rm -rf output/saved_simulations
```
