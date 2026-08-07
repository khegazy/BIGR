# external_artifacts

Files that BIGR needs at run time but that are **not part of the BIGR source tree**. They were
recovered from archived copies of the original SLAC/LCLS environment and vendored here so the
repository is self-contained — nothing outside the repo has to exist for the NO₂ analysis to run.

See `how_to_run.md` at the repo root for how these are wired in.

## Provenance

| File | Recovered from | Size | Used by |
|---|---|---|---|
| `modules/fitting.py` | `slac_archive/baseTools.zip` → `cds/home/k/khegazy/baseTools/modules/fitting.py` | 11 KB | `fit_legendres_images`, called at `modules/density_extraction.py:2891` (StoN error model only) |
| `modules/diffraction_simulation.py` | `slac_archive/simulations.zip` → `cds/home/k/khegazy/simulation/diffractionSimulation/modules/diffraction_simulation.py` | 15 KB | `diffraction_calculation`, called at `modules/density_extraction.py:2852` and `:2861` (StoN error model only) |
| `scattering_amplitudes/3.7MeV/*_dcs.dat` | `slac_archive/cds/home/k/khegazy/BIGR/scattering_amplitudes/3.7MeV/` | 96 KB | `get_scattering_amplitudes`, `modules/NO2.py:646`. **NO₂ needs `nitrogen_dcs.dat` and `oxygen_dcs.dat`**; the others (carbon, oxygen, fluorine, bromine, iodine) are included for other molecules |

### Why *these* copies

Several older `fitting.py` copies exist in the archive. Only this one defines
`fit_legendres_images` with the `image_stds=` and `chiSq_fit=` keyword arguments that
`density_extraction.py:2891` actually passes; the others would raise `TypeError`. The
`diffraction_simulation.py` copy here was likewise checked to expose
`diffraction_calculation(LMK, LMK_weights, R, atom_types, scat_amps, q_map, detector_dist=1, freq=None)`,
matching the call site exactly.

## Modifications made after vendoring

`modules/diffraction_simulation.py` was written against numpy < 1.24 and scipy < 1.15 and has
been modernised in place. Every change is logged in the "Code changes" table of
`how_to_run.md`. `modules/fitting.py` needed no changes.

## Scattering amplitude file format

ELSEPA-style differential-cross-section tables, parsed by fixed column slices (not whitespace
splitting) at `modules/NO2.py:646-664`:

- the first **31 lines are skipped** as header,
- `line[2:11]` → scattering angle in degrees,
- `line[39:50]` → differential cross section.

The reader converts these to `q = 4π sin(θ/2) / λ` and returns `sqrt(dcs)` as a cubic
interpolator, so the analysis `dom` must lie inside the tabulated `q` range. Filenames are
`<full element name>_dcs.dat`, where the name comes from the `atom_info` table at
`modules/density_extraction.py:476-484` (note it spells fluorine "flourine").

## What still works if this directory is deleted

The imports of `fitting` and `diffraction_simulation` are lazy (inside `simulate_error_StoN`),
so without this directory you can still run:

- the `("constant_sigma", σ)` and `("constant_background", σ)` error models,
- `mode_search.py`,
- all results/plotting paths in `NO2/analyze_results.ipynb`.

You cannot run the `("StoN", …)` error model, and you lose the scattering amplitudes, which
*are* required by every path — so you would need to supply your own `*_dcs.dat` files.
