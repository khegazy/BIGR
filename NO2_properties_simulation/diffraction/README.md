# Diffraction simulation — not included, and not needed to run BIGR

**Nothing in this folder is used by the NO₂ retrieval.** `build_posterior.py` and `mode_search.py`
do not touch it. If you are here to run the analysis, go back to
[`how_to_run.md`](../../how_to_run.md).

This folder holds the *output* of a separate diffraction simulator that was used to produce the
anisotropy basis plots (`plots/basis_*.png`) and the `test*.png` figures. **The simulator itself is
not part of this repository.**

`run_diffraction.sh` calls `./src/diffraction.py`, and `src` used to be a symlink to
`/media/kareem/…/simulations/diffractionSimulation` on an external drive. It was committed to git as
a 170-byte shortcut blob rather than a working symlink, so it has never resolved for anyone cloning
this repository. That blob has been removed, because a broken pointer that looks like a directory is
worse than an honest absence.

`run_diffraction.sh` also passes `--scat_amps_path /media/kareem/…/scatteringAmplitudes/3.7MeV/`.
The equivalent files *are* vendored in this repository at
[`external_artifacts/scattering_amplitudes/3.7MeV/`](../../external_artifacts/scattering_amplitudes/3.7MeV/),
so if you do restore the simulator, point that argument there instead.

## To use this again

1. Obtain the `diffractionSimulation` package (upstream:
   `github.com/khegazy/physics_simulations`, `diffractionSimulation/`).
2. Place or symlink it at `src/` inside this folder.
3. Edit `run_diffraction.sh` to point `--scat_amps_path` at
   `../../external_artifacts/scattering_amplitudes/3.7MeV/`.

Note that a copy of one module from that package — `diffraction_simulation.py` — *is* vendored at
`external_artifacts/modules/`, because the `StoN` error model imports it. That single module is not
the whole simulator and does not provide `diffraction.py`.
