import sys, os, glob, time
import argparse
import time
import h5py
from copy import copy as copy
import numpy as np
import scipy as sp
import corner
import numpy.random as rnd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, Process, Manager
import matplotlib.pyplot as plt
from matplotlib import cm, lines

from parameters import data_parameters
from modules.density_extraction import density_extraction, calc_all_dists


parser = argparse.ArgumentParser()
parser.add_argument("--do_ensemble", type=int, default=1, required=False)
parser.add_argument("--do_2dof", type=int, default=0, required=False)
parser.add_argument("--multiProc_ind", type=int, default=None, required=False)
args = parser.parse_args()


def log_prior(theta):
    d1_min, d1_max = 1e-4, 10.0
    d2_min, d2_max = 1e-4, 10.0
    d3 = np.sqrt((theta[:,0] - theta[:,1]*np.cos(theta[:,2]))**2
        + (theta[:,1]*np.sin(theta[:,2]))**2)
    log_prob = np.zeros(theta.shape[0])
    inds = (theta[:,2] > np.pi) + (theta[:,2] < 1e-3)\
          + (theta[:,0] < d1_min) + (theta[:,0] > d1_max)\
          + (theta[:,1] < d2_min) + (theta[:,1] > d2_max)
          #+ (theta[:,1] > d3) + (theta[:,0] > d3)           #

    log_prob[inds] = -1*np.inf

    return log_prob


def log_prior_2dof(theta):
  log_prob = np.zeros(theta.shape[0])
  inds = (theta[:,1] > np.pi) + (theta[:,1] < 1.)
  log_prob[inds] = -1*np.inf

  return log_prob


def theta_to_cartesian(theta):
    st, ct = np.sin(theta[:,:,2]/2), np.cos(theta[:,:,2]/2)
    molecules = np.zeros((theta.shape[0], theta.shape[1], 3, 3))
    molecules[:,:,0,0] = theta[:,:,0]*ct
    molecules[:,:,0,2] = theta[:,:,0]*st
    molecules[:,:,2,0] = theta[:,:,1]*ct
    molecules[:,:,2,2] = -1*theta[:,:,1]*st

    return molecules


def theta_to_cartesian_2dof(theta):
  st, ct = np.sin(theta[:,1]/2), np.cos(theta[:,1]/2)
  molecules = np.zeros((theta.shape[0], 3, 3))
  molecules[:,0,0] = theta[:,0]*ct
  molecules[:,0,2] = theta[:,0]*st
  molecules[:,2,0] = theta[:,0]*ct
  molecules[:,2,2] = -1*theta[:,0]*st

  return molecules


def initialize_walkers(params, molecule):
    #d1 = np.linalg.norm(molecule[0] - molecule[1])
    #d2 = np.linalg.norm(molecule[2] - molecule[1])
    #ang = np.arccos(np.sum(
    #    (molecule[0] - molecule[1])*(molecule[2] - molecule[1]))/(d1*d2))
    dists = rnd.normal(-1, 1,
        size=(params["Nwalkers"], len(params["sim_thetas"])-1))

    dists *= 0.05*np.expand_dims(np.array(params["sim_thetas"])[:-1], 0)
    dists += np.expand_dims(np.array(params["sim_thetas"])[:-1], 0)

    return dists 


def initialize_walkers_2dof(params, molecule):
  on1 = np.linalg.norm(molecule[0] - molecule[1])
  ang = np.arccos(np.sum(
      (molecule[0] - molecule[1])*(molecule[2] - molecule[1]))/(on1**2))
  dists = rnd.normal(0, 1, size=(params["Nwalkers"], 2))

  dists *= 0.01*np.array([[on1, ang]])
  dists += np.array([[on1, ang]])

  return dists


std_r = 0.02
std_a = 0.01
def Prob(x, m, s):
    return np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))

def molecule_ensemble_generator(thetas):
   
    d1, std_1 = thetas[:,0], thetas[:,1] 
    d2, std_2 = thetas[:,2], thetas[:,3]
    ang, std_a = thetas[:,4], thetas[:,5]
    if thetas.shape[-1] == 7:
      N = int(thetas[0,6])
    else:
      N = 7

    d1_distribution_vals = np.linspace(d1-std_1*4, d1+std_1*4, N)
    d2_distribution_vals = np.linspace(d2-std_2*4, d2+std_2*4, N)
    ang_distribution_vals = np.linspace(ang-std_a*4, ang+std_a*4, N)

    d1_probs = Prob(d1_distribution_vals, d1, std_r)
    d1_probs /= np.sum(d1_probs)
    d2_probs = Prob(d2_distribution_vals, d2, std_r)
    d2_probs /= np.sum(d2_probs)
    ang_probs = Prob(ang_distribution_vals, ang, std_a)
    ang_probs /= np.sum(ang_probs)

    ntt_1 = np.tile(np.expand_dims(d1_distribution_vals.transpose(), -1), (1,1,N**2))
    ntt_1 = np.reshape(ntt_1, (thetas.shape[0],-1))
    joint_probs = np.tile(np.expand_dims(d1_probs.transpose(), -1), (1,1,N**2))
    joint_probs = np.reshape(joint_probs, (thetas.shape[0], -1))

    ntt_2 = np.tile(np.expand_dims(d2_distribution_vals.transpose(), -1), (1,1,N))
    ntt_2 = np.tile(ntt_2, (1,N,1))
    ntt_2 = np.reshape(ntt_2, (thetas.shape[0],-1))
    joint_probs = joint_probs*np.reshape(
        np.tile(
          np.tile(np.expand_dims(d2_probs.transpose(), -1), (1,1,N)),
          (1,N,1)), (thetas.shape[0], -1))

    ntt_3 = np.tile(ang_distribution_vals.transpose(),(1,N**2))
    joint_probs = joint_probs*np.tile(ang_probs.transpose(), (1,N**2))

    new_thetas = np.concatenate([
        np.expand_dims(ntt_1, -1),
        np.expand_dims(ntt_2, -1),
        np.expand_dims(ntt_3, -1)], -1)
    del ntt_1, ntt_2, ntt_3

    molecules = theta_to_cartesian(new_thetas)
    joint_probs /= np.sum(joint_probs, -1, keepdims=True)
    
    if thetas.shape[0] == 1:
        mean_thetas = np.sum(
            new_thetas*np.expand_dims(joint_probs[0], -1), 1)
        mean_cartesian = np.sum(molecules\
            *np.expand_dims(np.expand_dims(joint_probs, -1), -1), 1)
        mean_d3 = np.sum(joint_probs\
            *np.linalg.norm(molecules[:,:,0] - molecules[:,:,-1], axis=-1), 1)
        std_thetas = np.sqrt(
            np.sum((new_thetas - mean_thetas)**2\
              *np.expand_dims(joint_probs, -1), 1))
        std_cartesian = np.sqrt(
            np.sum(np.expand_dims(np.expand_dims(joint_probs, -1), -1)\
              *(molecules - mean_cartesian)**2, 1))
        std_d3 = np.sqrt(np.sum(joint_probs\
            *(np.linalg.norm(molecules[:,:,0]-molecules[:,:,-1], axis=-1)\
              -mean_d3)**2, 1))

        print("\n#####  Mean molecule thetas  #####")
        print(mean_thetas[0])
        print("\n#####  STD molecule thetas  #####")
        print(std_thetas[0])
        print("\n#####  Mean molecule cartesian  #####")
        print(mean_cartesian[0])
        print("\n#####  STD molecule cartesian  #####")
        print(std_cartesian[0])
        print("\n#####  Mean d3  #####")
        print(mean_d3[0])
        print("\n#####  STD d3  #####")
        print(std_d3[0])
        print("\n")

    #print(np.sum(molecules*np.expand_dims(weights, -1), 0))
    return molecules, joint_probs


def molecule_ensemble_generator_randomGen(molecule):
    N = int(2e3)
    d1 = np.linalg.norm(molecule[0] - molecule[1])
    d2 = np.linalg.norm(molecule[2] - molecule[1])
    ang = np.arccos(np.sum(
        (molecule[0] - molecule[1])*(molecule[2] - molecule[1]))/(d1*d2))

    thetas = rnd.normal(0, 1, size=(N,3))*np.array([std_r, std_r, std_a])
    thetas = thetas + np.array([d1, d2, ang])
    print(np.array([d1, d2, ang]))
    print(thetas[:5,:])
    joint_probs = np.ones((N,1))*1./N

    molecules = theta_to_cartesian(thetas)
    norm = 1#np.sum(joint_probs)
    mean_thetas = np.sum(thetas*joint_probs, 0)/norm
    mean_cartesian = np.sum(np.expand_dims(joint_probs, -1)*molecules, 0)/norm
    mean_d3 = np.linalg.norm(np.sum((molecules[:,0] - molecules[:,-1])*joint_probs, 0)/norm)
    std_thetas = np.sqrt(np.sum((thetas - mean_thetas)**2*joint_probs, 0)/norm)
    std_cartesian = np.sqrt(np.sum(np.expand_dims(joint_probs, -1)*(molecules - mean_cartesian)**2, 0)/norm)
    std_d3 = np.sqrt(
      np.sum(joint_probs[:,0]*(np.linalg.norm(molecules[:,0]-molecules[:,-1], axis=-1)-mean_d3)**2, 0)/norm)

    print("\n#####  Mean molecule thetas  #####")
    print(mean_thetas)
    print("\n#####  STD molecule thetas  #####")
    print(std_thetas)
    print("\n#####  Mean molecule cartesian  #####")
    print(mean_cartesian)
    print("\n#####  STD molecule cartesian  #####")
    print(std_cartesian)
    print("\n#####  Mean d3  #####")
    print(mean_d3)
    print("\n#####  STD d3  #####")
    print(std_d3)
    print("\n")

    #print(np.sum(molecules*np.expand_dims(weights, -1), 0))
    return molecules, joint_probs


def single_molecule_generator(molecule):
  return np.expand_dims(molecule, 0), np.ones((1,1))


def get_NO2_ADMs(
    data_LMK, kwargs={},
    subtract_mean=False, normalize=False):

  bases_sub_dir = "A/temp-{}K".format(int(kwargs["temperature"]))
  folders = [kwargs["folder"], "NO2", bases_sub_dir]
  if "sub_dir" in kwargs:
    folders.append(kwargs["sub_dir"])
  folderName = os.path.join(*folders)
  files = "{}/A-Mat_NO2_{}K_Jm40 I={}TW tau=100fs D*.npy".format(
      folderName, int(kwargs["temperature"]), int(kwargs["intensity"]))

  print(folderName)
  print(files)
  fName = folderName.replace("/A/", "/times/").replace("temp-", "temp_")\
      + ".npy"
  print("time name", fName)
  with open(fName, "rb") as file:
    times_inp = np.load(file)


  # Get array of which bases to gather
  LMK = []
  basisList = []
  normsList = []

  # Gather bases
  for fl in glob.glob(files):
    D = fl[fl.find(" D")+2:-4]
    ln = len(D) + len(D)%2
    L = int(D[:ln//2])
    K = int(D[ln//2:])
    LMK.append(np.array([L,0,K], dtype=int))

    # Get the basis and normalize
    with open(fl, "rb") as file:
      basis = np.load(file)

    basis_interp = interp1d(times_inp, basis, kind='cubic')
    if "probe_FWHM" in kwargs:
      sz = times_inp.shape[0]*3
      times_interp = times_inp[0]\
          + np.arange(sz)*(times_inp[-1]-times_inp[0])/(sz-1)
      basis_interp = interp1d(times_inp, basis, kind='cubic')
      basis = basis_interp(times_interp)
      times = times_interp

      delta_time = times[1] - times[0]
      basis = gaussian_filter1d(basis,
          (0.001*kwargs["probe_FWHM"]/2.355)/delta_time)
      basis_interp = interp1d(times, basis, kind='cubic')

    basis = basis_interp(kwargs["eval_times"])

    basisList.append(basis)

    if subtract_mean:
      if L != 0:
        basisList[-1] -= np.mean(basisList[-1])
    #print("norm",L,np.sqrt(np.sum(bases[L]**2)), np.amax(np.abs(bases[L])))
    normsList.append(np.sqrt(np.sum(
        (basisList[-1] - np.mean(basisList[-1]))**2)))
    if normalize:
      basisList[-1] -= np.mean(basisList[-1])
      basisList[-1] /= normsList[-1]

  LMK    = np.array(LMK)
  allBases = np.array(basisList)
  allNorms = np.array(normsList)

  fit_bases = []
  for lmk_ in data_LMK:
    lInds = LMK[:,0] == lmk_[0]
    mInds = LMK[:,2] == np.abs(lmk_[2])

    fit_bases.append(allBases[lInds*mInds])

  return np.concatenate(fit_bases, axis=0)

##########################################################
def get_scattering_amplitudes(data_parameters, atom_types, atom_info):

  scat_amps_interp = {}
  for atm in atom_types:
    if atm in scat_amps_interp:
      continue

    angStr = []
    sctStr = []
    fName = os.path.join(data_parameters["scat_amps_path"],
        atom_info[atm][0] + "_dcs.dat")
    with open(fName, 'r') as inpFile:
      ind=0
      for line in inpFile:
        if ind < 31:
          ind += 1
          continue

        angStr.append(line[2:11])
        sctStr.append(line[39:50])

    angs = np.array(angStr).astype(np.float64)*np.pi/180
    q = 4*np.pi*np.sin(angs/2.)/data_parameters["wavelength"]
    scts = np.sqrt(np.array(sctStr).astype(np.float64))

    scat_amps_interp[atm] = interp1d(q, scts, 'cubic')

  return scat_amps_interp


def main():

  ###  Setup log prior and theta to cartesian  ###
  if args.do_2dof:
    input_log_prior = log_prior_2dof
    input_theta_to_cartesian = theta_to_cartesian_2dof
    input_initialize_walkers = initialize_walkers_2dof
  else:
    input_log_prior = log_prior
    input_theta_to_cartesian = theta_to_cartesian
    input_initialize_walkers = initialize_walkers
 
  ###  Setup ensemble generator  ###
  if args.do_ensemble:
    input_ensemble_generator = molecule_ensemble_generator
  else:
    input_ensemble_generator = single_molecule_generator


  ####################################
  #####  Run Geometry Retrevial  #####
  ####################################

  extraction = density_extraction(data_parameters,
      get_scattering_amplitudes,
      log_prior=input_log_prior,
      theta_to_cartesian=input_theta_to_cartesian,
      ensemble_generator=input_ensemble_generator,
      get_ADMs=get_NO2_ADMs)

  molecule = extraction.setup_calculations()

  """
  th = np.expand_dims(np.array([1.803-0.05, 1.118-0.05, 1.695-0.02]), 0)
  extraction.log_likelihood_density(th)
  sys.exit(0)
  """

  walkers_init = input_initialize_walkers(
      data_parameters, extraction.atom_positions)

  extraction.run_mcmc(walkers_init, data_parameters["run_limit"])



if __name__ == "__main__":
  q_max = [5, 7.5, 10, 15, 20]
  sigmas = [0.05, 0.1, 0.163, 0.5]
  #sg = sigmas[0]
  bkgs = [10**-4, 10**-6, 10**-8, 10**-10]
  ston = [300, 100, 33, 400]
  ston = [10]
  options = []
  for bg in sigmas:
    for lg in np.arange(4)+1:
      FB = []
      lmk = np.arange(lg+1)*2
      for l in lmk:
        if l == 0 or l%2 != 0:
          continue
        for k in lmk:
          if l == lmk[-1] and k != 0:
            continue
          if k <= l:
            FB.append(np.array([l,0,k]))
          #  FB.append(np.array([l,0,-1*k]))
          #elif k <= l:
          #  FB.append(np.array([l,0,k]))
      FB = np.array(FB).astype(int)

      for q in q_max:
        fit_range = [0.01, q]
        options.append({
          "fit_range" : fit_range,
          "dom"       : np.linspace(0, q, int(500*(1+fit_range[0]/fit_range[1]))),
          "fit_bases" : copy(FB),
          #"simulate_error" : ("StoN", (bg, [0.5,4]))})
          #"simulate_error" : ("constant_background", bg)})
          "simulate_error" : ("constant_sigma", bg)})


  print("LENNNNNNNnnnn", len(options))
  if args.multiProc_ind is not None:
    for k,v in options[args.multiProc_ind].items():
      data_parameters[k] = v
  if not args.do_ensemble:
    data_parameters["molecule"] = data_parameters["molecule"] + "_single"

  print("INP FB", data_parameters["fit_bases"])
  main()
