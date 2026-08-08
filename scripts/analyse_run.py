"""Summarise a finished MCMC run: convergence diagnostics and retrieved vs. ground truth.

Reads only the emcee backend HDF5 written by ``build_posterior.py`` (and the mode-search file
if present), so it needs no ADMs, scattering amplitudes or re-simulation.

Usage, from the repo root:
    .venv/bin/python scripts/analyse_run.py
    .venv/bin/python scripts/analyse_run.py --backend path/to/results_*.h5
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_NAMES = ["<NO(1)>", "sig(NO(1))", "<NO(2)>", "sig(NO(2))", "<ONO>", "sig(ONO)"]
DELTA_NAMES = ["<NO(1)>", "<NO(2)>", "<ONO>"]
PDF_TRUTH = np.array([1.35, 0.03, 1.05, 0.02, 2.34, 0.01])
DELTA_TRUTH = np.array([1.35, 1.05, 2.34])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=None, help="path to the results_*.h5 backend")
    args = ap.parse_args()

    if args.backend:
        backend = args.backend
    else:
        cands = [f for f in glob.glob(os.path.join(REPO, "output", "**", "results_*.h5"),
                                      recursive=True)
                 if "saved_simulations" not in f and "mode_search" not in os.path.basename(f)]
        if not cands:
            print("ERROR: no backend found under output/", file=sys.stderr)
            return 1
        backend = max(cands, key=os.path.getmtime)

    print("backend: {}\n".format(os.path.relpath(backend, REPO)))
    with h5py.File(backend, "r") as h5:
        chain = h5["chain"][:]                    # [steps, walkers, ndim]
        logp = h5["log_prob"][:]
        tau = h5["autocorr_times"][:]
        tau_hist = h5["tau_convergence"][:]
        accepted = h5["accepted"][:]
        converged = bool(h5["has_converged"][()])

    n_steps, n_walk, ndim = chain.shape
    names = PDF_NAMES if ndim == 6 else DELTA_NAMES
    truth = PDF_TRUTH if ndim == 6 else DELTA_TRUTH
    units = (["Ang"]*4 + ["rad"]*2) if ndim == 6 else ["Ang", "Ang", "rad"]

    # ---- convergence diagnostics ----
    print("=== convergence ===")
    print("  steps x walkers x ndim : {} x {} x {}".format(n_steps, n_walk, ndim))
    print("  has_converged          : {}".format(converged))
    print("  tau                    : {}".format(np.round(tau, 1)))
    print("  chain length / max tau : {:.1f}   (need > 4 for a non-empty thinned chain,"
          " > 50 for a trustworthy tau)".format(n_steps/np.nanmax(tau)))
    print("  acceptance fraction    : mean {:.3f}  min {:.3f}  max {:.3f}".format(
        (accepted/n_steps).mean(), (accepted/n_steps).min(), (accepted/n_steps).max()))

    # Does tau plateau, or does it track the chain length? The latter means the chain never
    # equilibrated, and the convergence test iteration > 100*tau can then never be satisfied.
    if len(tau_hist) >= 3:
        steps = np.arange(1, len(tau_hist)+1)*(n_steps/len(tau_hist))
        ratio = np.nanmean(tau_hist, axis=1)/steps
        print("  tau / step, first->last: {:.3f} -> {:.3f}".format(ratio[0], ratio[-1]))
        print("  verdict                : {}".format(
            "tau tracks chain length -> NOT equilibrated"
            if ratio[-1] > 0.5*ratio[0] else "tau is flattening -> equilibrating"))

    # ---- retrieved parameters ----
    mx = int(np.nanmax(tau))
    thin = chain[3*mx::mx].reshape(-1, ndim) if n_steps > 4*mx else np.empty((0, ndim))
    allp = chain[n_steps//3:].reshape(-1, ndim)          # post burn-in, correlated

    mode_file = os.path.join(os.path.dirname(backend),
                             "mode_search_" + os.path.basename(backend))
    modes = np.full(ndim, np.nan)
    if os.path.exists(mode_file):
        with h5py.File(mode_file, "r") as h5:
            modes = h5["ths_mean"][:]
    else:
        print("\n  (no mode-search file yet: run mode_search.py for Theta*)")

    print("\n=== retrieved vs ground truth ===")
    print("  independent samples: {}   correlated samples: {}".format(
        thin.shape[0], allp.shape[0]))
    print("\n  {:<12}{:>10}{:>11}{:>11}{:>12}{:>9}{:>7}".format(
        "parameter", "truth", "median", "sigma^Theta", "mode Theta*", "dev/sig", "unit"))
    print("  " + "-"*72)
    src = thin if thin.shape[0] >= 50 else allp
    for i in range(ndim):
        med, sig = np.median(src[:, i]), src[:, i].std()
        dev = (med - truth[i])/sig if sig > 0 else np.nan
        print("  {:<12}{:>10.5f}{:>11.5f}{:>11.5f}{:>12.5f}{:>9.2f}{:>7}".format(
            names[i], truth[i], med, sig, modes[i], dev, units[i]))
    print("\n  dev/sig is (median - truth)/sigma^Theta: |dev/sig| < 1 means truth is"
          " recovered within the posterior width.")
    print("  log_prob: first {:.4g}  last {:.4g}  best {:.4g}".format(
        logp[0].mean(), logp[-1].mean(), logp.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
