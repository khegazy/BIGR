import sys, os, glob, time
import argparse
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

from modules.density_extraction import density_extraction


########################
#####  Log Priors  #####
########################

def log_prior_3dof_gauss(theta):
  """
  Calculates the log prior for the normal distribution with 3 degrees of freedom by
  removing entries with unphysical values by setting the log prob to -infinity

      Parameters
      ----------
      theta : 2D np.array of type float [walkers,thetas]
          model parameters to be evaluated

      Returns
      -------
      log_prior : 1D np.array of type float [walkers]
          The log prior for each set of parameters
  """

  sig_max = 0.5
  d1_min, d1_max = 1e-4, 10.0
  d2_min, d2_max = 1e-4, 10.0

  log_prob = np.zeros(theta.shape[0])
  inds = (theta[:,0] < theta[:,2])\
        + (theta[:,4] > np.pi) + (theta[:,4] < 1e-3)\
        + (theta[:,0] < d1_min) + (theta[:,0] > d1_max)\
        + (theta[:,2] < d2_min) + (theta[:,2] > d2_max)\
        + (theta[:,1] < 0) + (theta[:,3] < 0) + (theta[:,5] < 0)\
        + (theta[:,1] > sig_max) + (theta[:,3] > sig_max) + (theta[:,5] > sig_max)
        #+ (theta[:,1] > d3) + (theta[:,0] > d3)           #

  log_prob[inds] = -1*np.inf

  return log_prob


def log_prior_3dof_delta(theta):
  """
  Calculates the log prior for the delta distribution with 3 degrees of freedom by
  removing entries with unphysical values by setting the log prob to -infinity

      Parameters
      ----------
      theta : 2D np.array of type float [walkers,thetas]
          model parameters to be evaluated

      Returns
      -------
      log_prior : 1D np.array of type float [walkers]
          The log prior for each set of parameters
  """

  d1_min, d1_max = 1e-4, 10.0
  d2_min, d2_max = 1e-4, 10.0

  #d3 = np.sqrt((theta[:,0] - theta[:,1]*np.cos(theta[:,2]))**2
  #    + (theta[:,1]*np.sin(theta[:,2]))**2)
  log_prob = np.zeros(theta.shape[0])
  inds = (theta[:,0] < theta[:,1])\
        + (theta[:,2] > np.pi) + (theta[:,2] < 1e-3)\
        + (theta[:,0] < d1_min) + (theta[:,0] > d1_max)\
        + (theta[:,1] < d2_min) + (theta[:,1] > d2_max)

  log_prob[inds] = -1*np.inf

  return log_prob


def log_prior_2dof_gauss(theta):
  """
  Calculates the log prior for the normal distribution with 2 degrees of freedom by
  removing entries with unphysical values by setting the log prob to -infinity

      Parameters
      ----------
      theta : 2D np.array of type float [N,thetas]
          model parameters to be evaluated

      Returns
      -------
      log_prior : 1D np.array of type float [N]
          The log prior for each set of parameters
  """

  sig_max = 0.5
  d1_min, d1_max = 1e-4, 10.0

  log_prob = np.zeros(theta.shape[0])
  inds = (theta[:,2] > np.pi) + (theta[:,2] < 0)\
        + (theta[:,0] < d1_min) + (theta[:,0] > d1_max)\
        + (theta[:,1] < 0) + (theta[:,3] < 0)\
        + (theta[:,1] > sig_max) + (theta[:,3] > sig_max)

  log_prob[inds] = -1*np.inf

  return log_prob


def log_prior_2dof_delta(theta):
  """
  Calculates the log prior for the delta distribution with 2 degrees of freedom by
  removing entries with unphysical values by setting the log prob to -infinity

      Parameters
      ----------
      theta : 2D np.array of type float [N,thetas]
          model parameters to be evaluated

      Returns
      -------
      log_prior : 1D np.array of type float [N]
          The log prior for each set of parameters
  """

  log_prob = np.zeros(theta.shape[0])
  inds = (theta[:,1] > np.pi) + (theta[:,1] < 1.)
  log_prob[inds] = -1*np.inf

  return log_prob



###############################################################
#####  Theta Parameters to Molecule Geometry (Cartesian)  #####
###############################################################

def theta_to_cartesian_single(theta):
  """
  Calculates the molecular cartesian coordinates for a list of 3 parameters

      Parameters
      ----------
      theta : 2D np.array of type float [N,dof]
          parameters to specify the molecular geometry

      Returns
      -------
      molecules : 3D np.array of type float [N,atoms,xyz]
          The cartesian coordinates of each atom in the molecule
  """

  st, ct = np.sin(theta[:,2]/2), np.cos(theta[:,2]/2)
  molecules = np.zeros((theta.shape[0], 3, 3))
  molecules[:,0,0] = theta[:,0]*ct
  molecules[:,0,2] = theta[:,0]*st
  molecules[:,2,0] = theta[:,1]*ct
  molecules[:,2,2] = -1*theta[:,1]*st

  return molecules


def theta_to_cartesian_ensemble(theta):
  """
  Calculates the molecular cartesian coordinates for N ensembles of 3 parameters

      Parameters
      ----------
      theta : 2D np.array of type float [N,ensemble_size,dof]
          parameters to specify the molecular geometry

      Returns
      -------
      molecules : 3D np.array of type float [N,ensemble_size,atoms,xyz]
          The cartesian coordinates of each atom in the molecule
  """

  st, ct = np.sin(theta[:,:,2]/2), np.cos(theta[:,:,2]/2)
  molecules = np.zeros((theta.shape[0], theta.shape[1], 3, 3))
  molecules[:,:,0,0] = theta[:,:,0]*ct
  molecules[:,:,0,2] = theta[:,:,0]*st
  molecules[:,:,2,0] = theta[:,:,1]*ct
  molecules[:,:,2,2] = -1*theta[:,:,1]*st

  return molecules


def theta_to_cartesian_2dof(theta):
  """
  Calculates the molecular cartesian coordinates for a list of 2 parameters

      Parameters
      ----------
      theta : 2D np.array of type float [N,dof]
          parameters to specify the molecular geometry

      Returns
      -------
      molecules : 3D np.array of type float [N,atoms,xyz]
          The cartesian coordinates of each atom in the molecule
  """

  st, ct = np.sin(theta[:,1]/2), np.cos(theta[:,1]/2)
  molecules = np.zeros((theta.shape[0], 3, 3))
  molecules[:,0,0] = theta[:,0]*ct
  molecules[:,0,2] = theta[:,0]*st
  molecules[:,2,0] = theta[:,0]*ct
  molecules[:,2,2] = -1*theta[:,0]*st

  return molecules


####################################
#####  Initialize MHA Walkers  #####
####################################

def initialize_walkers(params, molecule):
  """
  Return an initial set of theta parameters (walkers) to initialize the MCMC.

      Parameters
      ----------
      params : dictionary
          Runtime parameters from parameters.py containing initialization parameters
      molecule : 2D np.array of floats [atoms,xyz]
          Geometry of the groundstate in XYZ format

      Returns
      -------
      thetas : 2D np.array of floats [N,thetas]
          An array of initial guesses for the model parameters (theta)
  """

  thetas = rnd.normal(0, 1,
      size=(params["Nwalkers"], len(params["init_thetas"])))

  scale = 0.1
  if "init_thetas_std_scale" in params:
    scale = params["init_thetas_std_scale"]
  thetas *= scale*np.expand_dims(np.array(params["init_thetas"]), 0)
  thetas += np.expand_dims(np.array(params["init_thetas"]), 0)
  if len(params["init_thetas"]) == 4 or len(params["init_thetas"]) == 6:
    thetas[thetas[:,-2]>np.pi,-2] += np.pi - thetas[thetas[:,-2]>np.pi,-2]
  else:
    thetas[thetas[:,-1]>np.pi,-1] += np.pi - thetas[thetas[:,-1]>np.pi,-1]
  
  print("INFO: INITIAL WALKER MEAN/STD {} {}".format(
      np.mean(thetas, axis=0), np.std(thetas, axis=0)))

  return thetas


##########################################
#####  Molecular Ensemble Generator  #####
##########################################

def get_molecule_init_geo(params):
  """
  Import the ground state geometry in cartesian coordinates

      Parameters
      ----------
      params : dictionary
          Runtime parameters that specifies the location of the geometry

      Returns
      -------
      atom_types: list of strings [atoms]
          Element abbreviation of each atom
      atom_positions: 2D np.array of type float [atoms,xyz]
          An array of cartesian points for each atom
  """

  if not os.path.exists(params["init_geo_xyz"]):
    raise RuntimeError("Cannot find xyz file: " + params["init_geo_xyz"])

  atom_types      = []
  atom_positions  = []
  with open(params["init_geo_xyz"]) as file:
    for i,ln in enumerate(file):
      if i == 0:
        Natoms = int(ln)
      elif i > 1:
        vals = ln.split()
        print("\t {}: {} {} {}".format(*vals))
        atom_types.append(vals[0])
        pos = [float(x) for x in vals[1:]]
        atom_positions.append(np.array([pos]))

  return atom_types, np.concatenate(atom_positions, axis=0)



def Prob(x, m, s):
  """
  Calculate the Gaussian probability of observing x with mean m and std s.

      Parameters
      ----------
      x : N-D np.array of type float
          Point to evaluate the Gaussian distribution at
      m : N-D np.array of type float
          Mean of the Gaussian distribution
      s : N-D np.array of type float
          Standard deviation of the Gaussian distributio

      Returns
      -------
      y : N-D np.array of type float
          The log prior for each set of parameters
  """

  return np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))


def single_molecule_generator(thetas):      
  """
  Generate a single molecule geometry based on the delta function posterior
  from the given theta parameters

      Parameters
      ----------
      theta : 2D np.array of type float [N,thetas]
          Model parameters to be evaluated

      Returns
      -------
      molecules : 4D np.array of type float [N,1,atoms,xyz]
          The cartesian coordinates of the molecule
      probs : 2D np.array of type float [N, 1]
          The probability of each geometry, here set to 1
  """

  molecules = theta_to_cartesian_single(thetas)
  return np.expand_dims(molecules, 1), np.ones((thetas.shape[0],1))


def molecule_ensemble_generator(thetas):  
  """
  Generate an ensemble of molecular geometries based on the normal distribution
  posterior from the given theta parameters

      Parameters
      ----------
      theta : 2D np.array of type float [walkers,thetas]
          Model parameters to be evaluated

      Returns
      -------
      molecules : 4D np.array of type float [walkers,ensemble,atoms,xyz]
          The cartesian coordinates of the molecules
      probs : 2D np.array of type float [walkers,ensemble]
          The probability of each geometry to appear
  """

  # Get Values
  d1, std_1 = thetas[:,0], thetas[:,1] 
  d2, std_2 = thetas[:,2], thetas[:,3]
  ang, std_a = thetas[:,4], thetas[:,5]
  if thetas.shape[-1] == 7:
    N = int(thetas[0,6])
    thetas = thetas[:,:-1]
  else:
    N = 19
  std_ = 7

  d1_distribution_vals  = np.linspace(d1-std_1*std_, d1+std_1*std_, N)
  d2_distribution_vals  = np.linspace(d2-std_2*std_, d2+std_2*std_, N)
  ang_distribution_vals = np.linspace(ang-std_a*std_, ang+std_a*std_, N)

  #####  Setup Probabilities  #####
  d1_probs = Prob(d1_distribution_vals, d1, std_1)
  d1_probs /= np.sum(d1_probs)
  d2_probs = Prob(d2_distribution_vals, d2, std_2)
  d2_probs /= np.sum(d2_probs)
  ang_probs = Prob(ang_distribution_vals, ang, std_a)
  ang_probs /= np.sum(ang_probs)

  # Setup permutations of probabilities
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

  # Calculate molecular geometris and normalize probabilities
  molecules = theta_to_cartesian_ensemble(new_thetas)
  joint_probs /= np.sum(joint_probs, -1, keepdims=True)
  
  #####  Print Results  #####
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

      print("\n########################################")
      print("#####  Molecular Ensemble Results  #####")
      print("########################################")
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

  return molecules, joint_probs


"""
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

    molecules = theta_to_cartesian_ensemble(thetas)
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

"""

###########################################
#####  Get Axis Distribution Moments  #####
###########################################

def get_ADMs(params, get_LMK=None):
  """
  Imports the pre-calculated axis distributions moments (ADMs)

      Parameters
      ----------
      params : dictionary
          Runtime parameters specifying the ADMs' location
      get_LMK : 2d array-like of type int [N,LMK]
          Will return only the ADMs given here, otherwise returns all ADMs

      Returns
      -------
      LMK : 2D np.array of type int [N,LMK]
          The LMK values of each ADM
      bases : 2D np.array of type float [N,time]
          The imported ADMs
      norms : 1D np.array of type float [N]
          The norm of each ADM sqrt(sum(|ADM - mean(ADM)|^2))
      time : 1D np.array of type float [time]
          The evaluation times of the ADMs
  """

  print("INFO: Importing ADMs")
  if int(params["probe_FWHM"]*10) % 10 == 0:
    fwhm_str = str(int(params["probe_FWHM"]))
  else:
    fwhm_str = "{0:1g}".format(params["probe_FWHM"])
  if int(params["temperature"]*10) % 10 == 0:
    temp_str = str(int(params["temperature"]))
  else:
    temp_str = "{0:1g}".format(params["temperature"])

  folders = [params["folder"], "NO2", "ADMs",
      "temp-{}K".format(temp_str),
      "{}TW_{}fs".format(int(params["intensity"]), fwhm_str)]
  if "sub_dir" in params:
    folders.append(params["sub_dir"])
  folderName = os.path.join(*folders)
  files = os.path.join(folderName, "A*.npy")

  print("\tFolder: " + folderName)
  with open(os.path.join(folderName, "times.npy"), "rb") as file:
    times_inp = np.load(file)


  # Get array of which bases to gather
  LMK = []
  basisList = []
  normsList = []

  # Gather bases
  for fl in glob.glob(files):

    print("\t\tFile: " + fl)
    L = int(fl[-6])
    K = int(fl[-5])
    LMK.append(np.array([L,0,K], dtype=int))

    # Get the basis and normalize
    with open(fl, "rb") as file:
      basis = np.load(file)

    basis_interp = interp1d(times_inp, basis, kind='cubic')
    if "probe_FWHM" in params:
      sz = times_inp.shape[0]*3
      times_interp = times_inp[0]\
          + np.arange(sz)*(times_inp[-1]-times_inp[0])/(sz-1)
      basis_interp = interp1d(times_inp, basis, kind='cubic')
      basis = basis_interp(times_interp)
      times = times_interp

      delta_time = times[1] - times[0]
      basis = gaussian_filter1d(basis,
          (0.001*params["probe_FWHM"]/2.355)/delta_time)
      basis_interp = interp1d(times, basis, kind='cubic')

    basis = basis_interp(params["eval_times"])

    basisList.append(basis)

    if params["subtract_mean"]:
      if L != 0:
        basisList[-1] -= np.mean(basisList[-1])
    normsList.append(np.sqrt(np.sum(
        (basisList[-1] - np.mean(basisList[-1]))**2)))
    if params["normalize"]:
      basisList[-1] -= np.mean(basisList[-1])
      basisList[-1] /= normsList[-1]

  LMK    = np.array(LMK)
  allBases = np.array(basisList)
  allNorms = np.array(normsList)

  # Select desired LMK if get_LMK is not None
  fit_bases, fit_norms = [], []
  if get_LMK is not None:
    for lmk_ in get_LMK:
      lInds = LMK[:,0] == lmk_[0]
      mInds = LMK[:,2] == lmk_[2]

      fit_bases.append(allBases[lInds*mInds])
      fit_norms.append(allNorms[lInds*mInds])
    LMK = get_LMK
  else:
    fit_bases = allBases
    fit_norms = allNorms

  return LMK, np.concatenate(fit_bases, axis=0),\
      np.concatenate(fit_norms, axis=0), params["eval_times"]



###################################
#####  Scattering Amplitudes  #####
###################################

def get_scattering_amplitudes(params, atom_types, atom_info):
  """
  Import the atomic scattering amplitudes

      Parameters
      ----------
      params : dictionary
          Runtime parameters specifying the scattering amplitude location
      atom_types : list of strings [N]
          List of each atom's element abbreviation
      atom_info : dictionary { element : [name, amu mass]} 
          Information for all atoms

      Returns
      -------
      scat_amps_interp : dictionary { element : np.interp1d(scattering) }
          A dictionary of np.interp1d instances for the scattering amplitudes
  """

  scat_amps_interp = {}
  for atm in atom_types:
    if atm in scat_amps_interp:
      continue

    angStr = []
    sctStr = []
    fName = os.path.join(params["scat_amps_dir"],
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
    q = 4*np.pi*np.sin(angs/2.)/params["wavelength"]
    scts = np.sqrt(np.array(sctStr).astype(np.float64))

    scat_amps_interp[atm] = interp1d(q, scts, 'cubic')

  return scat_amps_interp



