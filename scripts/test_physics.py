"""Physics regression tests for the BIGR forward model.

There is no test suite in this repository, and the absence of one is why a centre-of-mass
bug that shrank every molecule by its total mass (46x for NO2) survived: it corrupted every
C coefficient, but nothing checked an invariant that would have exposed it.

Each test below asserts something that must hold regardless of parameters, so they stay
valid as the analysis configuration changes.

Run from the repo root:
    MPLBACKEND=Agg .venv/bin/python scripts/test_physics.py
Exits non-zero if any test fails.
"""

import contextlib
import os
import sys

import numpy as np
import scipy as sp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "NO2"))
os.chdir(os.path.join(REPO, "NO2"))          # the .so path is resolved relative to cwd

from parameters import get_parameters                                    # noqa: E402
from modules.density_extraction import density_extraction                # noqa: E402
import modules.NO2 as NO2mod                                             # noqa: E402
from modules.NO2 import (get_molecule_init_geo, get_scattering_amplitudes,  # noqa: E402
                        get_ADMs, log_prior_3dof_gauss,
                        molecule_ensemble_generator, single_molecule_generator)

TRUTH = np.array([1.35, 0.03, 1.05, 0.02, 2.34, 0.01])
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    print("  [{}] {}{}".format("PASS" if passed else "FAIL", name,
                               "  -- " + detail if detail else ""))
    return passed


def build(calc_type=0, grid_n=5):
    """Construct a density_extraction with a small ensemble grid, quietly."""
    NO2mod.ENSEMBLE_GRID_N = grid_n
    p = get_parameters()
    p["plot_setup"] = False
    p["save_sim_data"] = None
    p["simulate_error"] = ("constant_sigma", 1.0)
    p["calc_type"] = calc_type
    p["multiprocessing"] = 0
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        return density_extraction(
            p, get_molecule_init_geo, get_scattering_amplitudes,
            log_prior=log_prior_3dof_gauss,
            density_generator=molecule_ensemble_generator,
            ensemble_generator=molecule_ensemble_generator, get_ADMs=get_ADMs)


def pair_distances(geo, dist_inds):
    return np.array([np.linalg.norm(geo[a] - geo[b]) for a, b in zip(*dist_inds)])


def test_rotation_preserves_distances(ex):
    """A rotation into the molecular frame cannot change pairwise distances.

    This is the invariant the centre-of-mass bug violated: it scaled every coordinate by
    1/sum(mass), so all distances came out 46x too small for NO2.
    """
    mols, _ = single_molecule_generator(np.array([[1.35, 1.05, 2.34]]))
    before = pair_distances(mols[0, 0], ex.dist_inds)
    rot = ex.rotate_to_principalI_ensemble(mols.copy()).transpose((2, 3, 0, 1))
    after = ex.calculate_dists(rot)[:, 0, 0, 0]
    ok = np.allclose(np.sort(before), np.sort(after), rtol=1e-10)
    check("rotate_to_principalI_ensemble preserves pairwise distances", ok,
          "before {} vs after {}".format(np.round(before, 4), np.round(after, 4)))

    # the single-molecule variant must agree with the ensemble one
    rot1 = ex.rotate_to_principalI(mols[0, 0].copy())
    after1 = pair_distances(rot1, ex.dist_inds)
    check("rotate_to_principalI preserves pairwise distances",
          np.allclose(np.sort(before), np.sort(after1), rtol=1e-10),
          "after {}".format(np.round(after1, 4)))

    # and it must not mutate its input
    mols2, _ = single_molecule_generator(np.array([[1.35, 1.05, 2.34]]))
    snapshot = mols2.copy()
    ex.rotate_to_principalI_ensemble(mols2)
    check("rotate_to_principalI_ensemble does not mutate its input",
          np.array_equal(mols2, snapshot))


def test_backends_agree():
    """C++ and scipy backends must give the same C coefficients.

    calc_type=1 uses scipy.special.spherical_jn, which is independent of the C++ recursion,
    so agreement validates both the recursion and the surrounding combination logic.
    """
    th = TRUTH[None, :]
    NO2mod.ENSEMBLE_GRID_N = 5
    mols, w = molecule_ensemble_generator(th)
    ex0, ex1 = build(calc_type=0), build(calc_type=1)
    c_cpp = ex0.calculate_coeffs_ensemble_cpp(mols.copy(), w)[0]/ex0.I
    c_s0 = ex0.calculate_coeffs_ensemble_scipy(mols.copy(), w)[0]/ex0.I
    c_s1 = ex1.calculate_coeffs_ensemble_scipy(mols.copy(), w)[0]/ex1.I
    worst = 0.
    for i in range(len(ex0.data_LMK)):
        ref = np.sqrt(np.nanmean(c_cpp[i]**2))
        worst = max(worst,
                    np.nanmax(np.abs(c_s0[i]-c_cpp[i]))/max(ref, 1e-300),
                    np.nanmax(np.abs(c_s1[i]-c_cpp[i]))/max(ref, 1e-300))
    check("C++ / scipy-combination / scipy-Bessel backends agree", worst < 1e-8,
          "max relative deviation {:.2e}".format(worst))
    return ex0, c_cpp


def test_L_ordering(ex, c_cpp):
    """|C_400| must exceed |C_600|, and the coefficients must fall with L.

    At the q*dR probed, j_l is progressively suppressed with l, so higher L carries less
    signal. Paper Fig. 3c shows M_40k and M_60k roughly 2x and 4x smaller than M_20k.
    """
    idx = {(int(l[0]), int(l[2])): i for i, l in enumerate(ex.data_LMK)}
    rms = {k: np.sqrt(np.nanmean(c_cpp[i]**2)) for k, i in idx.items()}
    check("|C_400| > |C_600|", rms[(4, 0)] > rms[(6, 0)],
          "C400 {:.5g} vs C600 {:.5g}".format(rms[(4, 0)], rms[(6, 0)]))
    check("|C_200| > |C_400| > |C_600| (falls with L)",
          rms[(2, 0)] > rms[(4, 0)] > rms[(6, 0)],
          "{:.4g} > {:.4g} > {:.4g}".format(rms[(2, 0)], rms[(4, 0)], rms[(6, 0)]))
    # no coefficient should be orders of magnitude below its neighbours
    ratio = rms[(2, 0)]/max(rms[(4, 0)], 1e-300)
    check("C_200/C_400 is order unity, not orders of magnitude", ratio < 100,
          "ratio {:.4g}".format(ratio))


def test_spherical_harmonic_mapping():
    """sph_harm_y(n, m, polar, azim) must match the analytic Y_l^m.

    scipy removed sph_harm in 1.17 and sph_harm_y swaps both the degree/order and the
    angle order, so a silent mistake here would corrupt the physics without raising.
    """
    th, ph = 0.7, 1.3
    cases = [("Y_1^0", sp.special.sph_harm_y(1, 0, th, ph),
              np.sqrt(3/(4*np.pi))*np.cos(th)),
             ("Y_1^1", sp.special.sph_harm_y(1, 1, th, ph),
              -np.sqrt(3/(8*np.pi))*np.sin(th)*np.exp(1j*ph)),
             ("Y_2^0", sp.special.sph_harm_y(2, 0, th, ph),
              np.sqrt(5/(16*np.pi))*(3*np.cos(th)**2 - 1))]
    ok = all(np.isclose(got, want) for _, got, want in cases)
    check("sph_harm_y matches analytic Y_l^m", ok,
          "max |diff| {:.2e}".format(max(abs(g-w) for _, g, w in cases)))


def test_cpp_bessel_matches_scipy(ex):
    """The C++ even-order recursion must match scipy over the fitted q range."""
    x = ex.dom
    j = ex.spherical_j(x, len(x))
    worst, worst_l = 0., None
    for l in np.unique(ex.data_LMK[:, 0]):
        ref = sp.special.spherical_jn(int(l), x)
        dev = np.max(np.abs(j[int(l)//2] - ref))
        if dev > worst:
            worst, worst_l = dev, int(l)
    check("C++ spherical Bessel matches scipy over fit_range", worst < 1e-9,
          "worst |diff| {:.2e} at L={}".format(worst, worst_l))


def test_likelihood_peaks_at_truth():
    """The simulated data is the noiseless model at sim_thetas, so logL(truth) must be 0.

    Also checks that truth is a local maximum: every one-sided perturbation must lower it.
    """
    NO2mod.ENSEMBLE_GRID_N = 19          # the shipped grid; coarser grids are inaccurate
    ex = build(calc_type=0, grid_n=19)
    base = ex.log_likelihood(TRUTH[None, :])[0]
    check("logL(truth) == 0 (data is the noiseless model at sim_thetas)",
          abs(base) < 1e-9, "logL = {:.3g}".format(base))

    steps = np.array([0.02, 0.005, 0.02, 0.005, 0.02, 0.005])
    th = np.repeat(TRUTH[None, :], 12, axis=0)
    for i in range(6):
        th[2*i, i] += steps[i]
        th[2*i+1, i] -= steps[i]
    d = ex.log_likelihood(th) - base
    check("truth is a local maximum (all 12 perturbations lower logL)",
          np.all(d < 0), "worst (largest) delta = {:+.4g}".format(np.max(d)))


def test_measured_data_roundtrip(tmp_path="/tmp/bigr_measured_data_test.h5"):
    """Write an HDF5 in the documented measured-data layout and read it back.

    This is the format colleagues must produce for their own data (how_to_run.md section 10),
    so it is worth asserting that the documentation and the reader agree. It also exercises
    the branch that used to call the non-existent self.fig_I0.
    """
    import h5py
    ex = build(calc_type=0, grid_n=19)
    lmk, q, coeff = ex.data_LMK, ex.dom, ex.data_coeffs
    sigma = np.sqrt(ex.data_coeffs_var)

    with h5py.File(tmp_path, "w") as h5:
        h5["data_LMK"] = lmk.astype(int)
        h5["fit_axis"] = q
        for i in range(len(lmk)):
            h5["fit_LMK_dataLMKindex-{}".format(i)] = lmk[i][None, :].astype(int)
            h5["fit_coeffs_dataLMKindex-{}".format(i)] = coeff[i][:, None]
            h5["fit_coeffs_cov_dataLMKindex-{}".format(i)] = (sigma[i]**2)[:, None, None]

    p = get_parameters()
    p["plot_setup"] = False
    p["simulate_data"] = False
    p["data_fileName"] = tmp_path
    p["dom"] = None                  # adopt the file's fit_axis
    p["isMS"] = True
    p.pop("I_scale", None)           # force the fit_I0 branch (the old fig_I0 bug)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        ex2 = density_extraction(
            p, get_molecule_init_geo, get_scattering_amplitudes,
            log_prior=log_prior_3dof_gauss,
            density_generator=molecule_ensemble_generator,
            ensemble_generator=molecule_ensemble_generator, get_ADMs=get_ADMs)

    check("measured-data HDF5 round-trips through the documented layout",
          np.allclose(ex2.data_coeffs, coeff) and ex2.data_LMK.tolist() == lmk.tolist(),
          "max |diff| {:.2e}".format(np.nanmax(np.abs(ex2.data_coeffs - coeff))))
    check("measured data without I_scale fits the intensity (issue 006)",
          np.shape(ex2.I) == (1, 1) and np.isfinite(np.squeeze(ex2.I)),
          "fitted I = {:.6g}, shape {}".format(np.squeeze(ex2.I), np.shape(ex2.I)))
    os.remove(tmp_path)


def main():
    print("BIGR physics regression tests\n")
    print("rotation invariants:")
    ex = build(calc_type=0, grid_n=5)
    test_rotation_preserves_distances(ex)
    print("\nbackend agreement:")
    ex0, c_cpp = test_backends_agree()
    print("\nangular momentum ordering:")
    test_L_ordering(ex0, c_cpp)
    print("\nspecial functions:")
    test_spherical_harmonic_mapping()
    test_cpp_bessel_matches_scipy(ex0)
    print("\nlikelihood:")
    test_likelihood_peaks_at_truth()
    print("\nmeasured-data path:")
    test_measured_data_roundtrip()

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n{} passed, {} failed".format(len(RESULTS)-n_fail, n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
