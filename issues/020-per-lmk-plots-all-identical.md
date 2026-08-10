# 020 — Every per-LMK diagnostic plot showed the same (2,0,0) curve

**Severity** ~~P2~~
**Area** plotting / diagnostics
**Status** ✅ **FIXED 2026-08-08.** Both loops now index the data by the loop variable `il`.
Verified behaviourally: feeding a distinct row per LMK now yields six byte-distinct PNGs where it
previously yielded six identical ones. Guarded by
`scripts/test_physics.py::test_per_lmk_plots_are_distinct`.

## Symptom

All six `sim_add_data_noise_lmk-*.png` were byte-identical (same MD5), and so were all six
`high_pass_filter_lmk-*.png`:

```
MD5 (sim_add_data_noise_lmk-2-0-0.png) = a38c0d89f0512b78c4d3b54a675f777f
MD5 (sim_add_data_noise_lmk-2-0-2.png) = a38c0d89f0512b78c4d3b54a675f777f
MD5 (sim_add_data_noise_lmk-4-0-0.png) = a38c0d89f0512b78c4d3b54a675f777f
...                                       (all six the same, and likewise for high_pass_filter)
```

The filenames are correct and distinct; only the *contents* were wrong. Every file showed the
LMK = (2,0,0) row.

## Cause

The same defect in two places: the loop counter is used to build the filename but the data array is
indexed at a hardcoded `[0]`.

`simulate_error_data` (`modules/density_extraction.py:2851`):

```python
for il in range(self.input_data_coeffs.shape[0]):
    ...
    axs[0].plot(self.dom, self.input_data_coeffs[0,:], ...)     # <- [0], not [il]
    axs[1].plot(self.dom, self.experimental_noise[0,:], '-k')   # <- [0], not [il]
    ...
    fig.savefig(... "sim_add_data_noise_lmk-{}-{}-{}.png".format(*self.data_LMK[il]))  # il here
```

`plot_filter` (`:2911`):

```python
for il in range(self.data_LMK.shape[0]):
    ...
    ax.plot(fft_freqs[:imax+1], np.abs(fft_out[0,:imax+1])**2, '-k')   # <- [0], not [il]
    ...
    fig.savefig(... "high_pass_filter_lmk-{}-{}-{}.png".format(*self.data_LMK[il]))    # il here
```

## Impact

**Diagnostic only — no retrieved quantity is affected.** The noise actually added to the
coefficients is a whole-array operation (`:2849`, `self.input_data_coeffs += self.experimental_noise`)
and is correct; only the rendering of the per-LMK figures was wrong.

The cost was diagnostic blindness, and it is not negligible given this repository's history: these
plots are the intended way to check that the high-pass filter is isolating noise sensibly in each
angular channel, and that the added noise is comparable in scale to the coefficient it perturbs.
A per-LMK problem in the L = 4 or L = 6 channels — exactly the kind of defect that
[002](002-L4-coefficients-anomalously-small.md) turned out to be — would have been invisible here,
because those channels were never plotted.

`plot_filter` has a second caller at `:2354` that passes `axs=`, drawing into a multi-panel figure.
That figure had the same bug, so every panel showed the same curve.

## Note on the second call site

`plot_filter`'s `axs=` branch never closes or saves the figure itself (the caller owns it), and the
`axs is None` branch creates a figure per iteration without closing it. Not fixed here; it only
matters for LMK sets large enough to trip matplotlib's open-figure warning.

## Related

- [002](002-L4-coefficients-anomalously-small.md) — the class of per-L defect these plots exist to
  reveal, and which they could not have revealed.
