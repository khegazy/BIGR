import sys, os
import numpy as np
import numpy.random as rnd
from copy import copy
from collections import defaultdict
from matplotlib import pyplot as plt
from parameters import *

from modules.mode_search import *
from modules.density_extraction import density_extraction

from modules.NO2 import *


parser = argparse.ArgumentParser()
parser.add_argument("--do_ensemble", type=int, default=1, required=False)
parser.add_argument("--do_2dof", type=int, default=0, required=False)
parser.add_argument("--multiProc_ind", type=int, default=None, required=False)
args = parser.parse_args()


def main(data_params):
    """
    The main function that sets up the mode search and runs the mode search.
    The results are periodically saved, allowing one to kill and restart this
    function without losing progress. The data/simulation and 
    chi squared/likelihood calculations are handled by the density extraction
    class for consistency between this method and the posterior retrieval.

        Parameters
        ----------
        data_params : dictionary
            The dictionary of runtime parameters used to define variables
            for both the density extraction and mode search

        Returns
        -------
    """

    #####  Setup ensemble/density generators and log prior  #####
    if "density_model" in data_params:
        if data_params["density_model"] == "PDF":
            input_ensemble_generator = molecule_ensemble_generator
            input_density_generator = molecule_ensemble_generator
            if use_2dof or data_params["experiment"] == "2dof":
                input_log_prior = log_prior_2dof_gauss
            elif data_params["experiment"] == "3dof":
                input_log_prior = log_prior_3dof_gauss
            else:
                raise ValueError("Cannot handle experiment {}".format(
                        data_params["experiment"]))
        elif data_params["density_model"] == "delta":
            input_ensemble_generator = single_molecule_generator
            input_density_generator = single_molecule_generator
            if use_2dof or data_params["experiment"] == "2dof":
                input_log_prior = log_prior_2dof_delta
            elif data_params["experiment"] == "3dof":
                input_log_prior = log_prior_3dof_delta
            else:
                raise ValueError("Cannot handle experiment {}".format(
                        data_params["experiment"]))

        else:
            raise ValueError("Cannot handle density model {}".format(
                    data_params["density_model"]))
    else:
        raise ValueError("Must provide density_model parameter")

    extractor = density_extraction(data_params,
            get_molecule_init_geo,
            get_scattering_amplitudes,
            log_prior=input_log_prior,
            density_generator=input_density_generator,
            ensemble_generator=input_ensemble_generator,
            get_ADMs=get_ADMs,
            results_only=False)

    _, last_walker_pos = extractor.setup_sampler(
        data_params["Nwalkers"],
        len(data_params["init_thetas"]), expect_file=True)


    cs_fxn = get_cs_fxn(extractor)
    ths_dist, log_prob_dist =\
        extractor.get_mcmc_results(log_prob=True, plot=False)
    ths_dist_raw, log_prob_dist_raw =\
        extractor.get_mcmc_results(log_prob=True, plot=False, thin=False)

    #####  Run Mode Search  #####
    ths_start = get_starting_ths(data_params)
    N_ths = len(ths_start)


    log_prob_dist_raw, unique_inds = np.unique(
        log_prob_dist_raw, return_index=True)
    ths_dist_raw = ths_dist_raw[unique_inds]
    sort_inds = np.argsort(np.abs(log_prob_dist_raw))

    N_samples = data_params["N_mode_samples"]
    scale = np.exp(log_prob_dist_raw[sort_inds[:N_samples]]) + 1e-10 
    ths_mean = np.sum(ths_dist_raw[sort_inds[:N_samples]].transpose()*scale, 1)/np.sum(scale)
    ths_var = np.sum(((ths_dist_raw[sort_inds[:N_samples]] - ths_mean)**2).transpose()*scale, 1)\
        /np.sum(scale)
    ths_std = np.sqrt(ths_var)
    print("Result: {} +/- {}".format(ths_mean, np.sqrt(ths_var)))

    tic = time.time()
    ths_mean, ths_std, ths_sampled, log_prob_sampled, chiSq_sampled =\
        weight_avg_search(ths_dist_raw[sort_inds[:N_samples]].transpose(), 
            log_prob_dist_raw[sort_inds[:N_samples]], data_params,
            cs_fxn, data_params["mode_std_grid"], get_theta_search_input,
            extractor.get_fileName, tol=data_params["mode_tolerance"],
            N_best_ths=N_samples)
    print("INFO: Time to converge: {}".format((time.time() - tic)/60))


if __name__ == "__main__":
    
    #########################################################
    #####  Setup Method Parameters and Cluster Options  #####
    #########################################################

    data_parameters = get_parameters()
    use_2dof = False
    use_mol_ensemble = True

    #####  Change Parameters For Cluster Submissions  #####
    q_max = [10]
    #q_max = [5, 7.5, 10, 12.5, 15, 17.5, 20]
    #sigmas = np.insert(1./(10**np.arange(11)), 0, 0.163)
    ston = [100]
    #ston = [25, 50, 200, 400]
    lmk_arr = [[100, 100]]
    #lmk_arr = [[25, 12.5], [25, 25], [25, 50], [25, 100], [12.5, 100], [50, 100]]
    options = []

    for bg in ston:
      for lg in lmk_arr:
        adm_params = copy(data_parameters["ADM_params"])
        adm_params["temperature"] = lg[0]
        adm_params["probe_FWHM"] = lg[1]
        for q in q_max:
          fit_range = [0.5, q]
          options.append({
            "fit_range"  : fit_range,
            "dom"        : np.linspace(0, q, int(500*(1+fit_range[0]/fit_range[1]))),
            "ADM_params" : copy(adm_params),
            "simulate_error" : ("StoN", (bg, [0.5,4]))})
            #"simulate_error" : ("constant_background", bg)})
            #"simulate_error" : ("constant_sigma", bg)})

    # Select option if args.multiProc_ind is given
    if args.multiProc_ind is not None:
      for k,v in options[args.multiProc_ind].items():
        data_parameters[k] = v
      data_parameters = setup_dom(data_parameters)

    #############################
    #####  Run Mode Search  #####
    #############################
    main(data_parameters)

