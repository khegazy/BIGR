import sys, os
import h5py
import numpy as np
import numpy.random as rnd
from copy import copy
from collections import defaultdict
from matplotlib import pyplot as plt
from parameters import *
from parameters_data import get_parameters as get_parameters_data

from modules.density_extraction import density_extraction


def get_cs_fxn(extractor):
    """
    Creates the chi squared function implementation based on the log likelihood
    calculation from the chosen posterior in the density_extraction class.

        Parameters
        ----------
        extractor : density_extraction instance
            An instantiated instance of density_extraction with the parameters
            of the posterior one wants to find the mode of

        Returns
        -------
        cs_fxn : function
    """

    def cs_fxn(thetas):
        """
        Calculates the chi squared between the loaded data in extraction and
        input theta parameters using the log likelihood calculation in
        extraction

            Parameters
            ----------
            thetas : 2D np.array of type float [thetas,N]
                
        """
        thetas_rs = np.reshape(thetas, (thetas.shape[0], -1)).transpose()
        stride = 200
        chiSq = []
        for itr in range(thetas_rs.shape[0]//stride+1):
            if itr*stride < len(thetas_rs):
                #print("cs in iter", itr, thetas_rs[itr*stride:(itr+1)*stride,:].shape)
                chiSq.append(
                    -2*extractor.log_likelihood(
                      thetas_rs[itr*stride:(itr+1)*stride,:]))
                      
        # Take the mean over all LMK
        return np.reshape(np.concatenate(chiSq, axis=0), thetas.shape[1:])
    
    return cs_fxn


def get_starting_ths(data_params):
  """
  Get the starting point for the mode search

      Parameters
      ----------
      data_params : dictionary
          The dictionary of runtime parameters used when running the MCMC

      Returns
      -------
      init_thetas : 1D np.array of type float [thetas]
          The initial guess of the theta parameter values provided in
          data_parameters with key 'init_thetas'
  """

  if "init_thetas" not in data_params:
    raise RuntimeError("The parameter 'init_theta' must be specified in the"
        + "data parameters as the initial guess for the mode search and MCMC")

  return data_params["init_thetas"]


def get_theta_search_input(ths_best, ths_cluster_std, data_params):
    ths_range = np.zeros((ths_best.shape[0], 2))
    ths_range[:,0] = ths_best - ths_cluster_std*1.
    ths_range[:,1] = ths_best + ths_cluster_std*1.
    ths_range[ths_range<0] = 0.001
    return np.sum(ths_range, axis=1)/2, ths_range[:,1] - ths_range[:,0]


def convert_inds(a_min, shp):
  """
  Converts the 1 dimensional index a_min to the multi-dimensional index
  of shape shp

      Parameters
      ----------
      a_min : int
          The index of a single dimension that was flattened
      shp : 1D np.array of type int [N]
          The sizes of each of the N dimensions to unflatten to

      Returns
      -------
      indices : np.array of type int [N]
          The indices for each of the N dimensions corresponding to the
          flattened index a_min
  """

  cur_min = [a_min%shp[-1]]

  for i in range(1, len(shp)):
    cur_min.append((a_min%np.prod(shp[-1*(i+1):]))//np.prod(shp[-1*i:]))
  return np.flip(cur_min)


def setup_theta_scales_inds(std_steps, N_ths, N_inds, const_dims):

    def fill_b(ii,i):
        return i == 0 or not np.any(const_dims == ii)
    print("Nths", N_ths, N_inds)
    ths_scale = np.ones([N_ths]+N_inds)*np.nan
    ths_inds = np.ones([N_ths]+N_inds)
    #ths_inds = np.ones([N_ths-1]+[len(std_steps)]*(N_ths-1))
    ths_inds = ths_inds.astype(int)*-1
    for i in range(len(std_steps)):
        if fill_b(0,i):
            ths_scale[0,i] = std_steps[i]
            ths_inds[0,i] = i
        if fill_b(1,i):
            ths_scale[1,:,i] = std_steps[i]
            ths_inds[1,:,i] = i
        if fill_b(2,i):
            ths_scale[2,:,:,i] = std_steps[i]
            ths_inds[2,:,:,i] = i
        if N_ths > 3:
            if fill_b(3,i):
                ths_scale[3,:,:,:,i] = std_steps[i]
                ths_inds[3,:,:,:,i] = i
            if N_ths > 4:
                if fill_b(4,i):
                      ths_scale[4,:,:,:,:,i] = std_steps[i]
                      ths_inds[4,:,:,:,:,i] = i
                if N_ths > 5:
                    if fill_b(5,i):
                        ths_scale[5,:,:,:,:,:,i] = std_steps[i]
                        ths_inds[5,:,:,:,:,:,i] = i
                    if N_ths > 6:
                        if fill_b(6,i):
                            ths_scale[6,:,:,:,:,:,:,i] = std_steps[i]
                            ths_inds[6,:,:,:,:,:,:,i] = i
                        if N_ths > 7:
                            raise RuntimeError("Must fill for more thetas")
    
    return ths_inds, ths_scale

def save_mode_search(fileName,
        ths_mean_history, ths_mean, ths_std_history, ths_std, ths_var,
        ths_sampled, log_prob_sampled, chiSq_sampled):

    if ths_mean_history.shape[0] == 0:
        ths_mean_history = np.expand_dims(ths_mean, 0)
        ths_std_history = np.expand_dims(ths_std, 0)
    else:
        ths_mean_history = np.concatenate([ths_mean_history,
            np.expand_dims(ths_mean, 0)], axis=0)
        ths_std_history = np.concatenate([ths_std_history,
            np.expand_dims(ths_std, 0)], axis=0)
    with h5py.File(fileName, "w") as h5:
        h5.create_dataset("ths_mean", data=ths_mean)
        h5.create_dataset("ths_var", data=ths_var)
        h5.create_dataset("ths_std", data=ths_std)
        h5.create_dataset("ths_mean_history", data=np.array(ths_mean_history))
        h5.create_dataset("ths_std_history", data=np.array(ths_std_history))
        h5.create_dataset("ths_sampled", data=ths_sampled)
        h5.create_dataset("log_prob_sampled", data=log_prob_sampled)
        h5.create_dataset("chiSq_sampled", data=chiSq_sampled)


def weight_avg_search(ths_dist, log_prob_dist, data_params,
    cs_fxn, std_steps, get_theta_search, get_fileName,
    N_best_ths=25, tol=0.01, require_min_eval=False, verbose=False):

    #####  Find indices to search over and setup a grid search template  #####
    N_ths = ths_dist.shape[0]

    # With ths_width = np.ones(N_ths) will look at every dimension
    ths_width = np.ones(N_ths)
    N_inds, const_dims, dim_mask, dim_inds = [], [], [], []
    for i in range(N_ths):
        if ths_width[i] is None or np.isnan(ths_width[i]) or ths_width[i] == 0:
            ths_width[i] = 0
            N_inds.append(1)
            const_dims.append(i)
            dim_mask.append(False)
        else:
            N_inds.append(len(std_steps))
            dim_mask.append(True)
            dim_inds.append(i)
    for ii in const_dims:
        dim_inds.append(ii)
    dim_inds = np.array(dim_inds).astype(int)
    const_dims = np.array(const_dims)
    dim_mask = np.array(dim_mask)

    ths_inds, ths_scale = setup_theta_scales_inds(std_steps, N_ths, N_inds, const_dims)

    # Move const dims to the end and remove it's contribution to ths_scale
    if len(const_dims) and False:
        ths_scale = ths_scale.transpose()
        ths_inds = ths_inds.transpose()
        for i in range(len(const_dims)):
            #ths_scale = np.expand_dims(ths_scale[i], 0)
            ths_inds  = ths_inds[i]
        ths_scale = ths_scale.transpose()
        ths_inds  = ths_inds.transpose()

    #####  Setup Parameters and Import Previous Results  #####
    ths_sampled = copy(ths_dist)
    log_prob_sampled, chiSq_sampled = copy(log_prob_dist), -2*copy(log_prob_dist)
    perc_change, prev_ths_mean, ths_mean = np.ones(N_ths), np.zeros(N_ths), np.zeros(N_ths)
    ths_mean_history, ths_std_history = [], []
    convergence_count, switch_rnd = 0, 0
    fName = get_fileName()
    find = fName.rfind("/")
    fName = fName[:find+1] + "mode_search_" + fName[find+1:]
    fileName = os.path.join(data_params["output_dir"], fName+".h5")
    print("INFO: Looking for previous mode search in {}".format(fileName))
    ###  Retrieve Previous History  ###
    if os.path.exists(fileName):
        print("\tPrevious results were found")
        with h5py.File(fileName, "r") as h5:
            ths_mean = h5["ths_mean"][:]
            ths_var = h5["ths_var"][:]
            ths_std = h5["ths_std"][:]
            ths_mean_history = h5["ths_mean_history"][:]
            ths_std_history = h5["ths_std_history"][:]
            ths_sampled = h5["ths_sampled"][:]
            log_prob_sampled = h5["log_prob_sampled"][:]
            chiSq_sampled = h5["chiSq_sampled"][:]
        ths_mean_history = [copy(x) for x in ths_mean_history]
        ths_std_history = [copy(x) for x in ths_std_history]
        
        print("IMPORTED FROM h5")
        for i in -1*np.flip((2+np.arange(np.min([ths_mean.shape[0], 3]), dtype=int))):
            print("access", i, len(ths_mean_history),len(ths_mean_history))
            ths_mean, prev_ths_mean = ths_mean_history[i+1], ths_mean_history[i]
            perc_change = np.abs(1. - prev_ths_mean/ths_mean)
            if np.all(perc_change < 0.03):
                switch_rnd += 1
            else:
                switch_rnd = 0

            if np.all(perc_change < tol):
                convergence_count += 1
            else:
                convergence_count = 0

            print("test", i, ths_mean, prev_ths_mean, switch_rnd, perc_change)
        prev_ths_mean = ths_mean_history[-2]
    else:
        print("\tDid not find previous results, will now initialize")
    #ths_mean_history = np.array(ths_mean_history)
    #ths_std_history = np.array(ths_std_history)


    #####  Mode Search Loop  #####
    loop_count, same_val_count = 0, 0
    while convergence_count < 3 or np.all(prev_ths_mean==ths_mean):
        # Sort by most likely samples and drop low probability
        sort_inds = np.argsort(np.abs(log_prob_sampled))
        ths_dist = ths_sampled[:, sort_inds[:N_best_ths]]
        log_prob_dist = log_prob_sampled[sort_inds[:N_best_ths]]
        scale = np.exp(log_prob_dist) + 1e-10
        ths_mean = np.sum(
              np.reshape(ths_dist*scale, (N_ths, -1)), 1)/np.sum(scale)
        ths_var = np.sum(
              np.reshape(((ths_dist.transpose() - ths_mean)**2).transpose()\
                *scale, (N_ths, -1)), 1)/np.sum(scale)
        ths_std = np.sqrt(ths_var)
        ths_sampled = ths_sampled[:, sort_inds[:10000]]
        log_prob_sampled = log_prob_sampled[sort_inds[:10000]]
        chiSq_sampled = chiSq_sampled[sort_inds[:10000]]
        print("INFO: has weighted average {} +/- {} with converge_count {}".format(
            ths_mean, ths_std, convergence_count))

        #for i,j in zip(log_prob_dist, ths_dist.transpose()):
        #  print(i,j)
        # Save mode search progress
        if np.all(prev_ths_mean!=ths_mean) and loop_count > 0:
            ths_mean_history.append(copy(ths_mean))
            ths_std_history.append(copy(ths_std))
            save_mode_search(fileName,
                np.array(ths_mean_history), ths_mean, np.array(ths_std_history), ths_std, ths_var,
                ths_sampled, log_prob_sampled, chiSq_sampled)
            print("THETA MEAN SHAPE: ", len(ths_mean_history))

        # Check convergence criteria
        perc_change = np.abs(1. - prev_ths_mean/ths_mean)
        if np.all(perc_change < tol) and np.all(prev_ths_mean!=ths_mean):
            convergence_count += 1
            if convergence_count >= 3:
                break
        elif np.all(prev_ths_mean!=ths_mean):
            convergence_count = 0

        # Switch to grid or random search 
        ths_cent, ths_width = get_theta_search(ths_mean, ths_std, data_params)
        if np.all(prev_ths_mean!=ths_mean):
            if np.all(perc_change < 0.03):#np.any(perc_change > 0.05):
                switch_rnd += 1
            elif not (switch_rnd > 5 and loop_count-1 % 3 == 0):
                switch_rnd = 0

        if np.all(prev_ths_mean == ths_mean):
            same_val_count += 1
        else:
            same_val_count = 0
        ths_mask = np.zeros(1)

        # Make sure all the ths_eval are > 0 so as not to waste compute time
        # on unphysical parameters
        while np.sum(ths_mask) == 0:
            if switch_rnd < 5 or (same_val_count > 0 and same_val_count % 4 == 0):
                prand = []
                ths_eval = copy(ths_scale)
                for i in range(N_ths):
                    rnd_scale = 1
                    if np.all(prev_ths_mean == ths_mean):
                        rnd_scale = np.random.uniform() + 0.5
                    ths_eval[i] *= rnd_scale*ths_width[i]/2.
                    ths_eval[i] += ths_cent[i]
                    prand.append(copy(rnd_scale))
                print("random scaling", prand)
            else:
                rnd_scale = (0.5 + np.random.uniform()*1.5)
                ths_eval = np.random.normal(size=(1500,6))
                print("IN RANDOM THETA!!!", rnd_scale, ths_eval[0])
                ths_eval *= ths_width*rnd_scale/2.
                ths_eval += ths_cent
                ths_eval = ths_eval.transpose()
                #else:
                #    ind = np.argmax(ths_std)
                #    ths_eval = np.tile(ths_mean, (500, 1))
                #    ths_eval[:,ind] += np.linspace(-3.5, 3.5, 500)*ths_std[ind]
                #    ths_eval = ths_eval.transpose()

            ths_eval = np.reshape(ths_eval, (N_ths, -1))
            ths_mask = np.all(ths_eval > 0, axis=0)
            ths_eval = ths_eval[:,ths_mask]
            print("CHECK SHAPE", np.sum(ths_mask), ths_eval.shape)

        # Calculate the chi square for all the thetas
        chiSq = cs_fxn(ths_eval)
        min_inds = convert_inds(np.argmin(chiSq), chiSq.shape)
        min_chiSq = chiSq[tuple(min_inds)]
        ths_min = np.array([
            ths_eval[tuple(np.insert(min_inds, 0, i, axis=0))] for i in range(N_ths)])
        #print("\tINFO: Best chi squared of {} is at {}".format(min_chiSq, ths_min))

        ths_dist = copy(ths_eval)
        """
        for i,j in zip(chiSq/-2., ths_eval.transpose()):
          print(i,j)
        print("#############  prev highs  ###############")
        for i,j in zip(log_prob_dist, ths_dist.transpose()):
          print(i,j)
        """

        # Append results to history
        ths_sampled = np.concatenate([
            ths_sampled, np.reshape(ths_eval, (N_ths, -1))], -1)
        log_prob_sampled = np.concatenate([
            log_prob_sampled, np.reshape(chiSq, (-1))/-2], -1)
        chiSq_sampled = np.concatenate([
            chiSq_sampled, np.reshape(chiSq, (-1))], -1)

        prev_ths_mean = copy(ths_mean)
        loop_count += 1


    return ths_mean, ths_std, ths_sampled, log_prob_sampled, chiSq_sampled


########################
#####  DEPRECATED  #####
########################

def single_dim_search(ths_start, ths_step, cs_fxn,
    tol=0.01, require_min_eval=False, verbose=False):

    print("TEST MIN", cs_fxn(np.expand_dims(
        np.array([1.35, 0.03, 1.05, 0.02, 2.34, 0.01]), -1)))
    ths_min = ths_start
    cs_min = cs_fxn(np.expand_dims(ths_min, -1))[0]
    perc_change = np.ones_like(ths_min)
    while np.any(perc_change > tol):
        ths_min_prev = copy(ths_min)
        dim_order = np.arange(len(ths_min))
        np.random.shuffle(dim_order)
        for dim in dim_order:
            delta = ths_step[dim]
            ths_eval = copy(ths_min)
            ths_eval[dim] += delta
            cs_eval = cs_fxn(np.expand_dims(ths_eval, -1))[0]
            print("check dir", cs_min, cs_eval, ths_min, ths_eval)
            if cs_min < cs_eval:
                delta *= -1
                cs_eval = cs_min
                ths_eval = ths_min

            while cs_eval <= cs_min:
                ths_min = copy(ths_eval)
                cs_min = copy(cs_eval)
                ths_eval[dim] += delta
                cs_eval = cs_fxn(np.expand_dims(ths_eval, -1))[0]
            print("\tCompleted dim {} / {} at {}".format(dim, delta, ths_min[dim]))
        perc_change = np.abs(1. - ths_min_prev/ths_min)
        print("New Theta Min with {}: {} \ {}".format(cs_min, ths_min, perc_change))
    sys.exit(0)
    return cs_min, ths_min

