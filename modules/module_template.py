import sys, os, glob, time
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


##########################################
#####  Molecular Ensemble Generator  #####
##########################################

def single_molecule_generator(thetas, N=1):
  """
  Generate a single molecule geometry to calculate the expected signal

      Parameters
      ----------
      theta : 2D np.array of type float [walkers,thetas]
          model parameters to be evaluated

      Returns
      -------
      molecules : 4D np.array of type float [walkers,1,atoms,xyz]
          The cartesian coordinates of the molecule
      probs : 2D np.array of type float [walkers, 1]
          The probability of each geometry, here set to 1
  """

  raise NotImplementedError("The function 'single_molecule_generator' must be implemented")


def molecule_ensemble_generator(thetas, N=19):
  """
  Generate an ensemble of molecular geometries to calculate the expected signal

      Parameters
      ----------
      theta : 2D np.array of type float [walkers,thetas]
          model parameters to be evaluated

      Returns
      -------
      molecules : 4D np.array of type float [walkers,ensemble,atoms,xyz]
          The cartesian coordinates of the molecules
      probs : 2D np.array of type float [N, ensemble]
          The probability of each geometry to appear
  """
  
  raise NotImplementedError("The function 'molecule_ensemble_generator' must be implemented")


#######################################################
#####  MCMC Initialization/Probability Functions  #####
#######################################################

def log_prior_gauss(theta):
  """
  This calculates the log prior for the Gaussian distribution which will have 
  2*(3*N_atoms-6) theta parameters (a mean and standard deviation for each
  degree of freedom) to define the molecular geometry probability distribution.
  To remove only unphysical values and make every physical geometry equally 
  likely to visit, this function returns 0 for all physical values and
  -infinity for unphysical values e.g. negative distances/angles and 
  azimuthal/polar angles > 2pi/pi.

      Parameters
      ----------
      theta : 2D np.array of type float [walkers,thetas]
          model parameters to be evaluated

      Returns
      -------
      log_prior : 1D np.array of type float [walkers]
          The log prior for each set of parameters
  """

  raise NotImplementedError("The function 'log_prior_gauss' must be implemented")


def log_prior_delta(theta):
  """
  This calculates the log prior for the Delta distribution which will have 
  (3*N_atoms-6) theta parameters consisting of a mean for each degree of 
  freedom to define the molecular geometry probability distribution. To remove
  only unphysical values and make every physical geometry equally likely to
  visit, this function returns 0 for all physical values and -infinity for
  unphysical values e.g. negative distances/angles and
  azimuthal/polar angles > 2pi/pi.

      Parameters
      ----------
      theta : 2D np.array of type float [walkers,thetas]
          model parameters to be evaluated

      Returns
      -------
      log_prior : 1D np.array of type float [walkers]
          The log prior for each set of parameters
  """

  raise NotImplementedError("The function 'log_prior_delta' must be implemented")


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

  raise NotImplementedError("The function 'initialize_walkers' must be implemented")


################################################
#####  Importing Information/Calculations  #####
################################################

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

  raise NotImplementedError("The function 'get_molecule_init_geo' must be implemented")


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
  
  raise NotImplementedError("The function 'get_ADMs' must be implemented")


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

  raise NotImplementedError("The function 'get_scattering_amplitudes' must be implemented")
