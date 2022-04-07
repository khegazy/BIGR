import sys, os, glob, time
import errno
import h5py
import emcee
import corner
import numpy as np
import scipy as sp
import numpy.random as rnd
import multiprocessing as mp
from numpy.fft import fft, fftfreq
from copy import copy as copy
from functools import partial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from modules.spherical_j_cpp import spherical_j, calc_coeffs_cpp_helper

if os.path.exists("/cds/home/k/khegazy/simulation/diffractionSimulation/modules")
  sys.path.append("/cds/home/k/khegazy/simulation/diffractionSimulation/modules")
from diffraction_simulation import diffraction_calculation

if os.path.exists("/cds/home/k/khegazy/baseTools/modules")
  sys.path.append("/cds/home/k/khegazy/baseTools/modules")
from fitting import fit_legendres_images


def calc_dists(R):
  """
  Calculate the distances between all atoms in polar coordinates

      Parameters
      ----------
      R : 2D np.array of type float [atoms,xyz]
          The cartesian coordinates of each atom

      Returns
      -------
      diff : 3D np.array of type float [atoms,atoms,polar]
          A symmetric matrix of distances between atoms in polar coordinates
  """

  r     = np.expand_dims(R, 1) - np.expand_dims(R, 0)
  dR    = np.sqrt(np.sum(r**2, axis=2))
  theta = np.arccos(r[:,:,2]/(dR + 1e-20))
  phi   = np.arctan2(r[:,:,1], r[:,:,0])

  return np.concatenate([np.expand_dims(dR,2),\
    np.expand_dims(theta, 2),\
    np.expand_dims(phi, 2)], axis=2)


#def calc_all_dists(R):
def calc_ensemble_dists(R):
  """
  For each molecule in an ensemble we calculate the distances between
  all atoms in in a single molecule polar coordinates

      Parameters
      ----------
      R : 2D np.array of type float [N,atoms,xyz]
          The cartesian coordinates of each atom

      Returns
      -------
      diff : 3D np.array of type float [N,atoms,atoms,polar]
          A symmetric matrix of distances between atoms in polar coordinates
  """

  r     = np.expand_dims(R, 2) - np.expand_dims(R, 1)
  dR    = np.sqrt(np.sum(r**2, axis=-1))
  theta = np.arccos(r[:,:,:,2]/(dR + 1e-20))
  phi   = np.arctan2(r[:,:,:,1], r[:,:,:,0])

  return np.concatenate([np.expand_dims(dR,-1),\
    np.expand_dims(theta, -1),\
    np.expand_dims(phi, -1)], axis=-1)


class density_extraction:
  """
  This class retrieves the joint probability distribution (marginal likelihood)
  of theta (model) parameters. This class handles both importing data and
  simulating expected data with various options of error distributions.

      Attributes
      ----------
      data_params : dictionary
          Runtime parameters for this class including file paths and
          data/simulation parameters
      data_or_sim : bool
          If True one is evaluating data, if False one is simulating data with
          this package
      sampler : emcee.EnsembleSampler instance
          The emcee MCMC sampler that evaluates the Metropolis Hastings
          Algorithm
          
          has_converged : bool
          True if the MCMC has converged, False otherwise
      tau_convergence : list of np.array of type float
          Float of autocorrelation times throughout MCMC sampling
      atom_info : dictionary { element : [name,amu] }
          Dictionary of element information used for naming and calculating
          moments of inertia
      atom_types : list of strings
          A list of the element name for each atom in the molecule
      atom_positions : 2D np.array of type float [atoms,xyz]
          The cartesian coordinates of the molecule's ground state geometry
      dist_inds : tuple of 1D arrays of type int ([d],[d])
          The index of both atoms for the pair-wise distance d
      dist_sms_scat_amps : 3D np.array of type float [1,d,q]
          The scattering amplitude contribution in the sms representation
          for the pair-wise distance d
      I : 2D np.array of type float [1,1]
          The fit parameter for the intensity of the incoming diffraction probe
      do_multiprocessing : bool
          If True use multiprocessing to speed up C coefficient calculation
          in calculate_coeffs
      mp_manager : multiprocessing.Manager instance
          A multiprocessing class to manage the multiple cores when
          do_multiprocessing=True
      plot_setup : bool
          Plot this class' setup procedure if True, plot nothing if False


      Methods
      -------
      density_generator(thetas,N):
          Generates an ensemble of moleculer geometries parameterized by the
          given theta (model) parameters according to the posterior chosen 
          to approximate |Psi|^2
      ensemble_generator(thetas,N):
          Generates an ensemble of molecular geometries according to the |Psi|^2
          distribution that is known a priori and used to simulate input C
          coefficients and error bars
      setup_I_tensor_calc():
          Setup variables to calculate the moment of inertia tensor used to
          rotate molecules into the molecular frame
      evaluate_scattering_amplitudes():
          This function fills the scat_amps dictionary with interpolated
          scattering amplitudes evaluated at the reciprocal space measurement
          points and builds the atomic scattering.
      setup_calculations():
          This function sets up the spherical bessel function implementation used
          to calculate the C coefficients bases on the value of the runtime 
          parameter "calc_type"

          0 <- C++ implementation (Recommended, but cannot include very low q)
          1 <- Scipy implementation (Slowest but correct for all q values)
          2 <- Optimized Python implementation (Slower than 0 and same errors)

          If using option 0 or 2 one MUST check the "Checking Scale of Spherical
          Bessel Function Error" output and the check_jl{l}_calculations
          plots to make sure the residual is negligable for the data or 
          simulation's reciprocal space range.
      setup_sampler(nwalkers=None, ndim=None, expect_file=False):
          This function sets up the MCMC sampler used for the Metropolis Hastings
          Algorithm. It firsts imports previously saved backends and returns the 
          sampler to the previous state. If there is no previously saved file it 
          creates a new backend and sampler.
      prune_data():
          This function normalizes the C coefficients by the atomic scattering,
          if not already done, and removes data not used in the analysis. Below 
          are the runtime parameters that specify the data used for the analysis.

          isMS : bool
              True if the input data is already scaled by the atomic scattering
          fit_bases : 2D array-like [N,lmk]
              A list of all the LMK contributions used to calculate the likelihood
          fit_range : 1D array-like of type float [2]
              A list of the low and high range of reciprocal space used to
              calculate the likelihood
      calculate_I0(skip_scale=False, plot=True):
          Fits the input C200 contribution to the C200 calculation or uses the
          runtime parameter and plots the input and scaled calculation

          I_scale : float
              If None then fit C200, else use the given value
      calc_I_tensor(R):
          Calculate the moment of inertia tensor for an array of molecules
      calc_I_tensor_ensemble(R):
          Calculates the moment of inertia tensor for an array of 
          ensembles of molecules
      rotate_to_principalI(R):
          Rotate the molecular cartesian coordinates in R to the molecular frame
      rotate_to_principalI_ensemble(R):
          Rotate an ensemble molecular cartesian coordinates in R to the
          molecular frame
      fit_I0(calc_coeffs, data=None, var=None, return_vals=False):
          Fit the C200 coefficient for the intensity of the diffraction probe (I0)
          by minimizing the chi square
      simulate_error_data():
          This function introduces expiremental error, based on imported data,
          into the simulation by applying a high pass filter to the data and 
          adding this noise to the simulation, and using the imported variances
          from data.


      run_mcmc(walker_init_pos, Nsteps):
          This function runs the MCMC that evaluates the Metropolis Hastings
          Algorithm. It sets up the sampler, runs it in batches defined by a
          runtime parameter, and saves the results after each batch. It will 
          run for at least 100 autocorrelation times and once the autocorrelation
          time change is less than 1% have converged or the minimum number of 
          iterations has been set by the runtime parameter 'min_acTime_steps'.

          Nwalkers : int
              The number of MCMC walkers (chains)
          run_limit : int
              The number of MCMC steps within a single batch
          min_acTime_steps : int
              The minimum number of autocorrelation times until the MCMC can
              converge
      log_likelihood(theta):
          Calculate the log likelihood of with the theta parameters given the
          observed or simulated C coefficients by building an ensemble from the
          density generator
      log_prior(thetas):
          Calculate the log prior for the given thetas. Generally used to
          remove unphysical values
      default_log_prior(thetas):
          Returns a log prior of 0 indicating that all theta are equally
          likely to be selected
      C_log_likelihood(calc_coeffs):
          Calculate the log likelihood of the given C coefficients according to
          the error distribution from the measured C coeffcients
      log_probability(theta):
          Calculates and combines the log likelihood and the log prior for each
          set of theta (model) parameters later used to compare between theta
          parameters in the MCMC
      gaussian_log_likelihood(calc_coeffs):
          Calculate the log likelihood of the C coefficients when their error
          is Gaussian distributed
      calculate_coeffs_single_scipy(molecules):
          Calculate the C coefficients using Scipy functions for an array-like
          of molecular geometries in cartesian space
      calculate_coeffs_ensemble_scipy(R, w):
          Calculate the C coefficients using Scipy functions for an array-like
          of ensembles of molecular geometries in cartesian space
      calculate_coeffs_ensemble_cpp(R, weights, verbose=False):
          Calculate the C coefficients using the C++ implementation of the
          Spherical Bessel functions in this package
      calculate_coeffs_ensemble_multiProc(ensemble, weights):
          Split the C coefficient calculations among different processors based
          on the number specified in the runtime parameter

          multiprocessing : int
              The number of cores used to split the calculation
      calculate_coeffs_ensemble_multiProc_helper(procNum, R, weights, return_dict):
          A helper function to handle the return_dict when using multiprocessing

     
      get_mcmc_results(log_prob=False, labels=None, plot=True, thin=True):
          Load the MCMC results into the class and the previous state, as well
          as plot the correlations between theta parameters and the chain history
      get_fileName(folder_only=False, suffix=None):
          Create the file namd and/or address that uniquely defines the data or
          simulation based on the given runtime parameters that define this instance
      load_emcee_backend(fileName):
          Load the state and results (backend) of a previously saved emcee Sampler
          instances and print if the chain has converged or not
      save_emcee_backend():
          Saves the state and results (backend) of the current emcee Sampler
      def get_ADMs(params, get_LMK=None):
          Imports the pre-calculated axis distributions moments (ADMs)
      get_data():
          This function imports the C coefficents, their covariances, and labels
          from previously analysed, or calculated, data or simulation into internal
          class variables. Below are the expected keys and a description of the 
          data format in the h5 file. Each anisotropy component is indexed by the l.
          Here l corresponds to the L used in the LMK anisotropy contribution. So 
          each l contribution can have multiple fits from varying values of K and M.
          One will replace {l} by the index 0, 1, 2, ... and n corresponds to the
          different number of M and K variations fit to the data

          data_LMK : 2D np.array of type int [N,lmk]
              An array of all the LMK values in the file
          fit_LMK_dataLMKindex-{l} : 2D np.array of type int [n,lmk]
              The LMK values for the fits indexed by l
          fit_coeffs_dataLMKindex-{i} : 3D np.array of type [q,n]
              The C coefficients from fitting the l anisotropy components
          fit_coeffs_cov_dataLMKindex-{i} : 3D np.array of type [q,n,n]
              The C coefficient covariance matrix from fitting the l
              anisotropy components
          fit_axis : 1D np.array of type float [q]
              The reciprocal space values of the C coefficients
      simulate_data():
          This function will simulate expected C coefficients with different types
          of variations error distributions to apply this method in order to
          produce expected results from an experiment. To simulated data set the
          runtime parameters. 
          
          simulate_data : bool
              Will simulate and use C coefficients based on the ensemble from
              the ensemble generator if True
          simulate_error : tuple (string, params)
              Will simulate the C coefficient errors with the following options
                  Poissonian (Recommended) : ("StoN", (SNR, range))
                      Poissonian error is added to simulated diffraction images
                      and propagated into the C coefficients. They are normalized
                      such that the C200 signal to noise ratio calculated over
                      range (list) is SNR. 
                  Constant standard deviation : ("constant_sigma", sigma)
                      Sigma will be the standard deviation for all C coefficients
                  Constant background : ("constant_background", sigma)
                      Will apply a constant error to the C coefficient errors such
                      that the C200 signal to noise is given by sigma                
      simulate_error_StoN(error_options):
          This function simulates the C coefficient errors by adding Poissonian
          error to the diffraction pattern and propogating it through the fitting
          process.
      save_simulated_data():
          Save the simulated C coefficients and the calculated errors in an h5 
          file to use in the future to save time using the runtime parameter

          save_sim_data : string
              The folder to save the files in
      load_simulated_data():
          Load the simulated C coefficients and the calculated errors from the
          previosly saved h5 file the runtime parameter

          save_sim_data : string
              The folder to load the files in
      plot_emcee_results(samples, labels=None):
          Plots the the trajectory of each walker (chain) in the sampler, as well
          as the 2d projections of each pair of theta (model) parameters.
      plot_filter(data, plot_filter, plot_folder, axs=None):
          Plot the high pass and low pass filters applied in this analysis
  """
  
  data_params     = None
  sampler         = None
  has_converged   = False
  tau_convergence = None

  atom_info = {
      "H" : ["hydrogen", 1.0],
      "C" : ["carbon", 12.0],
      "O" : ["oxygen", 16.0],
      "N" : ["nitrogen", 14.0],
      "F" : ["flourine", 19.0],
      "Br": ["bromine", 78.0],
      "I" : ["iodine", 127.0]
  }

  I                   = np.ones((1,1))
  do_multiprocessing  = False
  plot_setup          = True


  density_generator   = None
  ensemble_generator  = None
  log_prior = None

  def __init__(self, data_params,
      get_molecule_init_geo,
      get_scattering_amplitudes,
      log_prior=None,
      density_generator=None,
      ensemble_generator=None,
      get_ADMs=None,
      results_only=False,
      make_plot_dirs=True):
    """
    This initialization function sets variables and functions, such as the
    log prior and molecular ensemble generators according to the input and
    imports scattering amplitudes and any other required files based on the
    input parameters. The data and error bars are either imported or simulated
    and pruned to only contain the data points used to retrieve the posterior.
    The error in the Spherical Bessel functions are also checked for the given
    data range and should be checked when using the C++ implementation.

        Parameters
        ----------
        data_params : dictionary
            All the runtime parameters used throughout the class to determine
            which functions to use, initial variable values, and addresses to
            folders and files.
        get_molecule_init_geo(params) : function
            Returns the ground state molecular geometry in cartesian
            coordinates before exciting the 
                Parameters
                ----------
                params : dictionary
                    Runtime parameters that specify the location of the file
                    containing the geometry

                Returns
                -------
                atom_types: list of strings [atoms]
                    Element abbreviation of each atom
                atom_positions: 2D np.array of type float [atoms,xyz]
                    An array of cartesian points for each atom
        log_prior(theta) : function
            Calculates the log prior for the normal distribution with 3 degrees
            of freedom by removing entries with unphysical values by setting 
            the log prob to -infinity

                Parameters
                ----------
                theta : 2D np.array of type float [walkers,thetas]
                    model parameters to be evaluated

                Returns
                -------
                log_prior : 1D np.array of type float [walkers]
                    The log prior for each set of parameters
        density_generator(thetas) : function
            Generate an ensemble of molecular geometries based on the chosen
            posterior from the given theta parameters to use in calculating 
            the likelihood

                Parameters
                ----------
                theta : 2D np.array of type float [walkers,thetas]
                    model parameters to be evaluated

                Returns
                -------
                molecules : 4D np.array of type float [walkers,ensemble,atoms,xyz]
                    The cartesian coordinates of the molecules
                probs : 2D np.array of type float [walkers,ensemble]
                    The probability of each geometry to appear
        density_generator(thetas) : function
            Generate an ensemble of molecular geometries based on the chosen
            posterior from the given theta parameters to use in calculating 
            the likelihood

                Parameters
                ----------
                theta : 2D np.array of type float [walkers,thetas]
                    model parameters to be evaluated

                Returns
                -------
                molecules : 4D np.array of type float [walkers,ensemble,atoms,xyz]
                    The cartesian coordinates of the molecules
                probs : 2D np.array of type float [walkers,ensemble]
                    The probability of each geometry to appear
        ensemble_generator(thetas) : function
            Generate an ensemble of molecular geometries based on the |Psi|^2
            distribution that is known a priori in order to calculate the 
            input C coefficients and error bars

                Parameters
                ----------
                theta : 2D np.array of type float [1,thetas]
                    model parameters to be evaluated

                Returns
                -------
                molecules : 4D np.array of type float [1,ensemble,atoms,xyz]
                    The cartesian coordinates of the molecules
                probs : 2D np.array of type float [1,ensemble]
                    The probability of each geometry to appear
        get_ADMs(params, get_LMK=None) : function
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
        results_only : boolean
            Set to True when instantiating this class to only import results
            to save time by not importing the MCMC backend. Default is False
        make_plot_dirs : boolean
            Set to False when only looking at results and not running the MCMC
            in order to not make plot folders in other folders or subfolders
    """

    self.data_params = data_params
    self.density_generator = density_generator
    self.ensemble_generator = ensemble_generator
    if "output_dir" not in self.data_params:
      self.data_params["output_dir"] = "output"


    # Log Prior
    if log_prior is not None:
      self.log_prior = log_prior
    else:
      self.log_prior = self.default_log_prior

    # Convert between MCMC parameters theta to cartesian
    #if theta_to_cartesian is not None:
    #  self.theta_to_cartesian = theta_to_cartesian

    # Gather ADMs for simulated error
    if get_ADMs is not None:
      self.get_ADMs = get_ADMs

    # Plotting
    if "plot_setup" in data_params:
      self.plot_setup = data_params["plot_setup"]

    # Skip if results only
    if results_only:
      fileName = os.path.join(self.data_params["output_dir"], self.get_fileName() + ".h5")
      self.load_emcee_backend(fileName)

      return

    # Make output folders
    print("INFO: Making output folder")
    if not os.path.exists(os.path.join(
        self.data_params["output_dir"], self.get_fileName(folder_only=True))):
      try:
        os.makedirs(os.path.join(
            self.data_params["output_dir"], self.get_fileName(folder_only=True)))
      except OSError as e:
        if e.errno != errno.EEXIST:
          raise
    if make_plot_dirs and not os.path.exists(
        os.path.join("plots", self.get_fileName(folder_only=True))):
      try:
        os.makedirs(os.path.join("plots", self.get_fileName(folder_only=True)))
      except OSError as e:
        if e.errno != errno.EEXIST:
          raise

    # Setup posterior type
    print("INFO: Setup Multiprocessing")
    self.C_log_likelihood = self.gaussian_log_likelihood
    if "multiprocessing" in self.data_params:
      if self.data_params["multiprocessing"] > 1:
        self.do_multiprocessing = True
        self.mp_manager = mp.Manager()
    else:
      self.data_params["multiprocessing"] = 1

    # Get initial geometry
    print("INFO: Importing ground state geometry")
    self.atom_types, self.atom_positions = get_molecule_init_geo(self.data_params)
    
    # Setup moment of inertia tensor calculation
    print("INFO: Setting up moment of inertia calculation")
    self.setup_I_tensor_calc()

    # Rotate initial geometry into the MF
    print("INFO: Rotating initial state molecular frame")
    self.atom_positions = self.rotate_to_principalI(self.atom_positions)

    # Get data
    print("INFO: Getting data")
    self.data_or_sim = True
    if "simulate_data" in self.data_params:
      if self.data_params["simulate_data"]:
        self.data_or_sim = False
    self.get_data()

    # Get scattering amplitudes
    print("INFO: Getting scattering amplitudes")
    self.scat_amps_interp = get_scattering_amplitudes(
        self.data_params, self.atom_types, self.atom_info)
    self.evaluate_scattering_amplitudes()
    
    dist_inds1 = []
    dist_inds2 = []
    self.dist_sms_scat_amps = []
    for i, a1 in enumerate(self.atom_types):
      for j_, a2 in enumerate(self.atom_types[i+1:]):
        j = j_ + i+1
        dist_inds1.append(i)
        dist_inds2.append(j)
        self.dist_sms_scat_amps.append(
            self.scat_amps[a1]*self.scat_amps[a2]/self.atm_scat)
    self.dist_inds = (np.array(dist_inds1), np.array(dist_inds2))
    self.dist_sms_scat_amps = np.expand_dims(
        np.array(self.dist_sms_scat_amps), axis=0)

    # Setup Spherical Bessel function and C coefficient calculation
    self.setup_calculations()

    # Simulate data if needed
    self.data_Lcalc = np.reshape(self.data_LMK[:,0], (-1, 1, 1))
    self.data_Mcalc = np.reshape(self.data_LMK[:,1], (-1, 1, 1))
    self.data_Kcalc = np.reshape(self.data_LMK[:,2], (-1, 1, 1))
    self.calc_dom = np.expand_dims(self.dom, axis=0)
    if not self.data_or_sim:
      print("INFO: Simulate data")
      self.simulate_data()

    # Prune data in time and dom
    print("INFO: Pruning data")
    self.prune_data()

    # Remove global offset
    """
    if "global_offset" in self.data_params:
      if self.data_params["global_offset"]:
        self.remove_global_offset()
    """

    # Calculate Wiener mask
    self.make_wiener_weight()

    
    # Calculate I0
    print("INFO: Calculating I0")
    self.calculate_I0()
    
    # Subtract the mean from the data
    # TODO Fix mean subtraction or whatever it should be | Same for C_dist
    #self.data_coeffs -= np.expand_dims(np.mean(
    #    self.data_coeffs[:,:], axis=-1), -1)


    """
    # Apply customized likelihood using data
    if "fit_likelihood" in self.data_params:
      if self.data_params["fit_likelihood"]:
        dim_shape = extraction.C_distributions.shape
        hists, bins = [], []
        Nbins=75
        for data in np.reshape(extraction.C_distributions,
            (dim_shape[0], -1)).transpose():

            h,b = np.histogram(data, bins=Nbins)
            hists.append(copy(h))
            bins.append((b[1:]+b[:-1])/2)
        hists = np.array(hists)
        bins = np.array(bins)
        
        pols = np.arange(10)
        self.fit_liklihood_offset = bins[:np.argmax(hists[-1])]
        X = [np.ones_like(np.expand_dims(bins))]
        for p in pols:
          X.append(X[-1]*np.expand_dims(bins - np.expand_dims(bins[:,off_dims],-1), 1))
        X = np.concatenate(X, 1)
        print(X.shape, np.linalg.inv(np.einsum('ikj,ilj->ikl', X, X)).shape)
        print(np.einsum('ij,ikj->ik', hists, X).shape)
        histss = copy(hists)
        histss[hists==0] = 1
        print(np.sum(np.isnan(np.log(histss))))
        coeffs = np.einsum('ikj,ij->ik',
            np.linalg.inv(np.einsum('ikj,ilj->ikl', X, X)),
              np.einsum('ij,ikj->ik', np.log(histss), X))

        def fit_log_likelihood(calc_coeffs):
    """

    ###  Perform Sanity Checks and Function Validations  ###
    print("###############################################################")
    print("#####  Checking Scale of Spherical Bessel Function Error  #####")
    print("###############################################################")
    j_check = self.spherical_j(np.reshape(self.dom, (1, -1, 1, 1)))
    for n in np.unique(self.data_LMK[:,0]):
      ii = np.where(self.data_LMK[:,0] == n)[0][0]
      std_check = np.abs(sp.special.spherical_jn(n, self.dom)-j_check[ii,0,:,0,0])\
          /np.sqrt(self.data_coeffs_var[ii])
      std_check[np.isnan(std_check)] = 0
      m = np.amax(std_check)
      if m < 1.:
        mm = "Passed"
      else:
        mm = "FAILED AT Q={}".format(self.dom[np.argmax(std_check)])
      print("L = {} \t Largest std = {} at index {} ... {}".format(
          n, m, np.argmax(std_check), mm))

      # Plot Comparison
      if self.data_params["plot_setup"]:
        fig, ax = plt.subplots()
        ax.plot(self.dom, j_check[ii,0,:,0,0], '-k')
        ax.plot(self.dom, sp.special.spherical_jn(n, self.dom), '--b')
        ax.set_ylabel("j$_{}$".format(n))
        ax.set_xlabel(r"q $[\AA^{-1}]$")
        ax.set_xlim(self.dom[0], self.dom[-1])
        ax1 = ax.twinx()
        ax1.plot(self.dom, std_check)
        ax1.set_ylabel("residual [std]")
        #ax1.plot(px, (res[i][0,cut:,0,0]-sp.special.spherical_jn(i*2, x[0,cut:,0,0]))/sp.special.spherical_jn(i*2, x[0,cut:,0,0]))
        fig.savefig("plots/check_jl{}_calculation.png".format(n))
    print("\n")


  def theta_to_cartesian(self, theta):
    """
    Converts MCMC parameters to cartesian coordinates

    Parameters
    ----------
    theta: 2D np.array of type float [walkers,theta]
        The theta (model) parameters

    Returns:
        Cartesian representation of each molecule [Nwalkers, Natoms, 3]
    """

    return theta


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

    raise NotImplementedError("Must implement get_ADMs")


  def setup_I_tensor_calc(self):
    """
    Setup variables to calculate the moment of inertia tensor used to rotate molecules
    into the molecular frame

        Parameters
        ----------

        Returns
        -------
    """

    self.mass = np.array([self.atom_info[t][1] for t in self.atom_types])
    self.mass = np.reshape(self.mass, (-1,1,1))
    self.Itn_inds1, self.Itn_inds2 = np.array([2,0,1]), np.array([1,2,0])
    self.Itn_idiag = np.arange(3)

  
  def calc_I_tensor(self, R):
    """
    Calculate the moment of inertia tensor for an array of molecules

        Parameters
        ----------
        R : np.array of type float
            An array-like of molecular cartesian coordinates

        Returns
        -------
    """

    # Off diagonal terms
    I_tensor = -1*np.expand_dims(R, -1)*np.expand_dims(R, -2)

    # Diagonal terms
    I_tensor[:,self.Itn_idiag,self.Itn_idiag] =\
        R[:,self.Itn_inds1]**2 + R[:,self.Itn_inds2]**2

    return np.sum(I_tensor*self.mass, 0)

  
  def calc_I_tensor_ensemble(self, R):
    """
    Calculates the moment of inertia tensor for an array of 
    ensembles of molecules

        Parameters
        ----------
        R : np.array of type float
            An array-like of ensembles of molecular cartesian coordinates

        Returns
        -------
    """

    # Off diagonal terms
    I_tensor = -1*np.expand_dims(R, -1)*np.expand_dims(R, -2)

    # Diagonal terms
    I_tensor[:,:,:,self.Itn_idiag,self.Itn_idiag] =\
        R[:,:,:,self.Itn_inds1]**2 + R[:,:,:,self.Itn_inds2]**2

    return np.sum(I_tensor*self.mass, -3)




  def get_data(self):
    """
    This function imports the C coefficents, their covariances, and labels
    from previously analysed, or calculated, data or simulation into internal
    class variables. Below are the expected keys and a description of the 
    data format in the h5 file. Each anisotropy component is indexed by the l.
    Here l corresponds to the L used in the LMK anisotropy contribution. So 
    each l contribution can have multiple fits from varying values of K and M.
    One will replace {l} by the index 0, 1, 2, ... and n corresponds to the
    different number of M and K variations fit to the data

    data_LMK : 2D np.array of type int [N,lmk]
        An array of all the LMK values in the file
    fit_LMK_dataLMKindex-{l} : 2D np.array of type int [n,lmk]
        The LMK values for the fits indexed by l
    fit_coeffs_dataLMKindex-{i} : 3D np.array of type [q,n]
        The C coefficients from fitting the l anisotropy components
    fit_coeffs_cov_dataLMKindex-{i} : 3D np.array of type [q,n,n]
        The C coefficient covariance matrix from fitting the l
        anisotropy components
    fit_axis : 1D np.array of type float [q]
        The reciprocal space values of the C coefficients

        Parameters
        ----------

        Returns
        -------
    """

    self.ADMs = None
    if "data_fileName" in self.data_params:
      print("looking at file: ", self.data_params["data_fileName"])
      with h5py.File(self.data_params["data_fileName"], "r") as h5:
        self.diffraction_LMK_ = h5["data_LMK"][:]
        self.data_LMK_, self.input_data_coeffs_var_ = [], []
        for i in range(self.diffraction_LMK_.shape[0]):
          self.data_LMK_.append(
              h5["fit_LMK_dataLMKindex-{}".format(i)][:].astype(int))
          cov_inds = np.arange(self.data_LMK_[-1].shape[0])
          self.input_data_coeffs_var_.append(np.transpose(
              h5["fit_coeffs_cov_dataLMKindex-{}".format(i)][:][:,cov_inds,cov_inds]))
        #self.input_data_coeffs_var *= np.expand_dims(self.input_data_coeffs_var[:,70,:,:], -1)
        #self.data_lg = h5["legendre_inds"][:]
        self.dom_ = h5["fit_axis"][:]*self.data_params["q_scale"]
        self.input_data_coeffs_ = []
        for i in range(self.diffraction_LMK_.shape[0]):
          self.input_data_coeffs_.append(np.transpose(
              h5["fit_coeffs_dataLMKindex-{}".format(i)][:]))
        self.input_data_coeffs_ = np.concatenate(self.input_data_coeffs_, axis=0)
      self.data_LMK_ = np.concatenate(self.data_LMK_, axis=0)
      self.input_data_coeffs_var_ = np.concatenate(self.input_data_coeffs_var_, axis=0)
      
      # Keep only lmk elements specified by fit_bases parameter
      if "fit_bases" in self.data_params:
        inds = []
        for lmk in self.data_params["fit_bases"]:
          for i,lmk_ in enumerate(self.data_LMK_):
            if np.all(lmk == lmk_):
              inds.append(i)
              break
        inds = np.array(inds).astype(int)
        self.data_LMK_ = self.data_params["fit_bases"]
        self.input_data_coeffs_ = self.input_data_coeffs_[inds]
        self.input_data_coeffs_var_ = self.input_data_coeffs_var_[inds]
    else:
      self.data_LMK_ = self.data_params["fit_bases"]
      self.input_data_coeffs_ = None
      self.input_data_coeffs_var_ = None
      self.dom_ = self.data_params["dom"]


    self.dom_mask = np.ones_like(self.dom_).astype(bool)
    if "fit_range" in self.data_params:
      self.data_params["fit_range"][1] =\
          np.min([self.data_params["fit_range"][1], self.dom_[-1]])
      """
      if self.data_params["fit_range"][1] > self.dom[-1]:
        delta = self.dom[1] - self.dom[0]
        Nsteps = np.ceil(
            (self.data_params["fit_range"][1] - (self.dom[-1] + delta))/delta)
        self.dom = np.concatenate([self.dom,
          np.linspace(self.dom[-1] + delta, self.dom[-1] + delta + Nsteps*delta,
            Nsteps+1)])
        self.dom_mask = np.ones_like(self.dom).astype(bool)
      """
      
      self.dom_mask[self.dom_<self.data_params["fit_range"][0]] = False
      self.dom_mask[self.dom_>self.data_params["fit_range"][1]] = False

    if self.data_or_sim:
      self.dom = self.dom_
      self.data_LMK = self.data_LMK_
      self.input_data_coeffs = self.input_data_coeffs_
      self.input_data_coeffs_var = self.input_data_coeffs_var_

      ind2 = np.where(
          (self.data_LMK[:,0] == 2) & (self.data_LMK[:,2] == 0))[0]
      SN_ratio_lg2 = np.nanmean(
          self.input_data_coeffs_[ind2,self.dom_mask]**2\
            /self.input_data_coeffs_var_[ind2,self.dom_mask])
      #print("SN data ratio L=2: {} {}".format(
      #    SN_ratio_lg2, self.data_params["fit_range"]))
    else:
      self.data_LMK = self.data_params["fit_bases"]
      self.dom = self.data_params["dom"]

    # Use dom from data when simulating data
    if self.data_params["dom"] is None or "ata" in self.data_params["dom"]:
      print("INFO: Using dom from data")
      self.dom = self.dom_

    if "simulate_error" in self.data_params:
      error_type, error_options = self.data_params["simulate_error"]
      if error_type == "data":
        self.dom = self.dom_


  def simulate_data(self):
    """
    This function will simulate expected C coefficients with different types
    of variations error distributions to apply this method in order to
    produce expected results from an experiment. To simulated data set the
    runtime parameters. 
    
    simulate_data : bool
        Will simulate and use C coefficients based on the ensemble from
        the ensemble generator if True
    simulate_error : tuple (string, params)
        Will simulate the C coefficient errors with the following options
            Poissonian (Recommended) : ("StoN", (SNR, range))
                Poissonian error is added to simulated diffraction images
                and propagated into the C coefficients. They are normalized
                such that the C200 signal to noise ratio calculated over
                range (list) is SNR. 
            Data : 
            Constant standard deviation : ("constant_sigma", sigma)
                Sigma will be the standard deviation for all C coefficients
            Constant background : ("constant_background", sigma)
                Will apply a constant error to the C coefficient errors such
                that the C200 signal to noise is given by sigma                
                
        Parameters
        ----------

        Returns
        -------
    """

    self.C_distributions = []
    
    # Check for error simulation type
    if "simulate_error" in self.data_params:
      error_type, error_options = self.data_params["simulate_error"]
    else:
      raise ValueError("Must specify data parameter 'simulate_error' when simulating data.")

    # Must use MS for data since simulations are in MS
    if error_type == "data":
      if not self.data_params["isMS"]:
        atm_scat_ = np.expand_dims(self.atm_scat, 0)
        self.input_data_coeffs_ /= atm_scat_
        self.input_data_coeffs_var_ /= atm_scat_**2

        self.data_params["isMS"] = True

      if np.any(self.dom != self.dom_):
        raise ValueError("When using data variance with simulated data one must use data dom.")

    #####  Load Data if it Exists  #####
    data_is_loaded = False
    if "save_sim_data" in self.data_params:
      if self.data_params["save_sim_data"]:
        data_is_loaded = self.load_simulated_data()

    if not data_is_loaded:
      #######################
      ###  Simulate Data  ###
      #######################

      if self.ensemble_generator is not None:
        # Get ensemble of molecules
        ensemble = self.ensemble_generator(
            np.expand_dims(np.array(self.data_params["sim_thetas"])[:-1], 0))
        if len(ensemble) == 2:
          ensemble, weights = ensemble
        else:
          weights = np.ones((ensemble.shape[0], 1))

        self.input_data_coeffs_var = None
        docompare = False
        self.input_data_coeffs =\
            self.calculate_coeffs(ensemble, weights)[0]
        if docompare:
          self.input_data_coeffs =\
              self.calculate_coeffs_ensemble_cpp(ensemble, weights)[0]
          print("TEST RES CPP",self.input_data_coeffs[:,100:130])
          self.input_data_coeffs = np.zeros(
              (self.data_LMK[:,0].shape[0], self.dom.shape[0]))
          ind, ind_step = 0, int(np.ceil(100000./ensemble.shape[0]))
          while ind < ensemble.shape[1]:
            ind_end = np.min([ensemble.shape[1], ind + ind_step])
            #calc_coeffs = self.calculate_coeffs_ensemble(ensemble)
            print("WS", weights.shape)
            calc_coeffs = self.calculate_coeffs_ensemble_scipy(
                ensemble[:,ind:ind_end], weights[:,ind:ind_end])
            self.input_data_coeffs = self.input_data_coeffs\
                + np.sum(np.sum(calc_coeffs\
                *np.expand_dims(np.expand_dims(weights[:,ind:ind_end], -1), -1),
                0), 0)
            self.C_distributions.append(calc_coeffs)
            ind = ind_end
     
          print("TEST RES PYT",self.input_data_coeffs[:,100:130])
          sys.exit(0)
                                    
        """
        ind, ind_step = 0, 10000
        while ind < ensemble.shape[0]:
          ind_end = np.min([ensemble.shape[0], ind + ind_step])
          self.input_data_coeffs_var = self.input_data_coeffs_var\
              + np.sum(
                (self.calculate_coeffs_ensemble(ensemble[ind:ind_end])\
                  - self.input_data_coeffs)**2\
                *np.expand_dims(weights[ind:ind_end], -1), 0)
          ind = ind_end
        """
        #print("DATA", self.input_data_coeffs[-2,2], self.input_data_coeffs_var[-2,2])
        self.data_params["isMS"] = True
        #self.C_distributions = np.transpose(
        #    np.concatenate(self.C_distributions, 1),
        #    (1,2,0))
        
        #plt.errorbar(self.dom, self.data_coeffs[i,:], np.sqrt(self.data_coeffs_var[i,:]))
      else:
        raise ValueError("Must specify ensemble generator when initializing class")
        

      ###########################
      ###  Simulate Variance  ###
      ###########################
      self.experimental_var = None
      plot_folder = os.path.join("plots", self.data_params["molecule"])
      
      ###  Generate errors for the total signal  ###
      if "StoN" in error_type:
        self.simulate_error_StoN(error_options)
      elif "data" in error_type or "Data" in error_type:
        self.simulate_error_data()
      elif "onstant_background" in error_type:
        variance_scale = error_options
        self.experimental_var = np.ones(self.input_data_coeffs.shape[-1], dtype=np.float)
        if self.data_params["isMS"]:
          self.experimental_var /= self.atm_scat**2
        
        # Error scale derived from input signal
        ind2 = (self.data_LMK[:,0] == 2)*(self.data_LMK[:,2] == 0)
        if self.input_data_coeffs_var is None:
          SN_ratio_lg2 = np.nanmean(
              self.input_data_coeffs[ind2,self.dom_mask]\
                /np.sqrt(self.experimental_var[self.dom_mask]))**2
        else:
          SN_ratio_lg2 = np.nanmean(
              self.input_data_coeffs[ind2,self.dom_mask]\
                /np.sqrt(self.input_data_coeffs_var[ind2,self.dom_mask]))**2
        #print("SCALE", self.data_params["simulate_error_scale"], SN_ratio_lg2/self.data_params["simulate_error_scale"])

        # Combine error scale with input scale factor
        if variance_scale is None:
          self.data_params["simulate_error"] = (error_type, 1.)
          variance_scale = 1.

        variance_scale = variance_scale**2*SN_ratio_lg2

        self.experimental_var *= variance_scale

      elif "onstant_sigma" in error_type:
        self.experimental_var =\
            np.ones_like(self.input_data_coeffs)*error_options**2
        if not self.data_params["isMS"]:
          self.experimental_var *= self.atm_scat**2
      
      else:
        raise RuntimeError("ERROR: Cannot handle error type " + error_type)



      ###  Propogate error through fits to generate error for each LMK  ###
      if "sigma" not in error_type and False:
        if self.ADMs is None:
          if "ADM_kwargs" in self.data_params:
            self.ADMs = self.get_ADMs(
                self.data_LMK, kwargs=self.data_params["ADM_kwargs"], normalize=False)
          else:
            self.ADMs = self.get_ADMs(self.data_LMK)


        fit_var = np.zeros_like(self.input_data_coeffs)
        for l in np.unique(self.data_LMK[:,0]):
          linds = (self.data_LMK[:,0] == l)
          print(self.data_LMK)
          print(self.data_LMK[:,0])
          print(self.ADMs.shape)
          fit_ADMs = self.ADMs[linds,:]
          fit_ADMs -= np.mean(fit_ADMs, -1, keepdims=True)
          #print("SSSS", self.input_data_coeffs.shape, self.input_data_coeffs_var.shape, self.experimental_var.shape)
          print("err shapes", self.ADMs.shape, fit_ADMs.shape, self.input_data_coeffs.shape)
          print(np.matmul(fit_ADMs, np.transpose(fit_ADMs)))
          #print(np.linalg.inv(np.matmul(fit_ADMs, np.transpose(fit_ADMs))))
          fit_var_ = np.linalg.inv(np.matmul(fit_ADMs, np.transpose(fit_ADMs)))
          print(fit_var_)
          fit_var_ = np.expand_dims(fit_var_, 0)*np.reshape(self.experimental_var, (-1, 1, 1))
          ev_inds = np.arange(fit_var_.shape[-1])
          fit_var[linds,:] = np.transpose(fit_var_[:,ev_inds,ev_inds])
        self.experimental_var = fit_var


      if self.input_data_coeffs_var is not None:
        self.input_data_coeffs_var += self.experimental_var
      else:
        self.input_data_coeffs_var = copy(self.experimental_var)

      print("FIN SHAPES", self.input_data_coeffs_var.shape, self.input_data_coeffs.shape)

      if "save_sim_data" in self.data_params:
        if self.data_params["save_sim_data"]:
          self.save_simulated_data()
   

    ind2 = (self.data_LMK[:,0] == 2)*(self.data_LMK[:,2] == 0)
    SN_ratio_lg2 = np.nanmean(
        self.input_data_coeffs[ind2,self.dom_mask]**2\
          /self.input_data_coeffs_var[ind2,self.dom_mask])
    #print("SN sim ratio L=2: {} {}".format(
    #    SN_ratio_lg2, self.data_params["fit_range"]))
    
    """
    tmp = copy(self.input_data_coeffs)
    #shift = rnd.normal(size=self.input_data_coeffs.shape)
    #shift *= np.sqrt(self.input_data_coeffs_var[:,:,0,0])
    # TODO debugging
    #self.input_data_coeffs += shift
    print(self.input_data_coeffs.shape, self.input_data_coeffs_var.shape)
    plt.plot(self.dom[70:], tmp[0,70:])
    plt.errorbar(self.dom[70:], self.input_data_coeffs[0,70:], np.sqrt(self.input_data_coeffs_var[0,70:]))
    plt.savefig("init_werr.png")
    plt.close()
    sys.exit(0)
    """
 
    if self.input_data_coeffs_var is None:
      raise RuntimeError("ERROR: input_data_coeffs_var is still None!!!")


  def save_simulated_data(self):
    """
    Save the simulated C coefficients and the calculated errors in an h5 
    file to use in the future to save time using the runtime parameter

    save_sim_data : string
        The folder to save the files in
    
        Parameters
        ----------

        Returns
        -------
    """

    if not os.path.exists(os.path.join(
        self.data_params["save_sim_data"], self.get_fileName(folder_only=True))):
      try:
        os.makedirs(os.path.join(
            self.data_params["save_sim_data"], self.get_fileName(folder_only=True)))
      except OSError as e:
        if e.errno != errno.EEXIST:
          raise

    fileName = os.path.join(
        self.data_params["save_sim_data"], self.get_fileName() + ".h5")

    print("INFO: Saving simulated data {}".format(fileName))

    with h5py.File(fileName, "w") as h5:
      h5.create_dataset("input_data_coeffs", data=self.input_data_coeffs)
      h5.create_dataset("input_data_coeffs_var", data=self.input_data_coeffs_var)
      if self.experimental_var is not None:
        h5.create_dataset("experimental_var", data=self.experimental_var)


  def load_simulated_data(self):
    """
    Load the simulated C coefficients and the calculated errors from the
    previosly saved h5 file the runtime parameter

    save_sim_data : string
        The folder to load the files in
    
        Parameters
        ----------

        Returns
        -------
    """

    fileName = os.path.join(
        self.data_params["save_sim_data"], self.get_fileName() + ".h5")

    if not os.path.exists(fileName):
      print("INFO: {} does not exist, now simulating data".format(fileName))
      return False
    else:
      print("INFO: {} exist, now loading data".format(fileName))
      
      with h5py.File(fileName, "r") as h5:
        self.input_data_coeffs = h5["input_data_coeffs"][:]
        self.input_data_coeffs_var = h5["input_data_coeffs_var"][:]
        if "experimental_var" in h5.keys():
          self.experimental_var = h5["experimental_var"][:]
        else:
          self.experimental_var = None
    
    # Simulated Data is in MS
    self.data_params["isMS"] = True

    return True


  def prune_data(self):
    """
    This function normalizes the C coefficients by the atomic scattering,
    if not already done, and removes data not used in the analysis. Below 
    are the runtime parameters that specify the data used for the analysis.

    isMS : bool
        True if the input data is already scaled by the atomic scattering
    fit_bases : 2D array-like [N,lmk]
        A list of all the LMK contributions used to calculate the likelihood
    fit_range : 1D array-like of type float [2]
        A list of the low and high range of reciprocal space used to
        calculate the likelihood

        Parameters
        ----------

        Returns
        -------
    """

    # Devide by atomic scattering
    if not self.data_params["isMS"]:
      atm_scat_ = np.expand_dims(self.atm_scat, 0)
      self.input_data_coeffs /= atm_scat_
      #atm_scat_ = np.expand_dims(atm_scat_, -1)
      self.input_data_coeffs_var /= atm_scat_**2
 
   
    # Prune the list of legendre projections
    if "fit_bases" in self.data_params:
      temp_LMK, temp_data_LMK = [], []
      temp_data_var_LMK, temp_C_distributions_LMK = [], []
      for lmk in self.data_params["fit_bases"]:
        ind = (self.data_LMK[:,0]==lmk[0])*(self.data_LMK[:,1]==lmk[1])\
            *(self.data_LMK[:,2]==lmk[2])
        if np.sum(ind) != 1:
          print("ERROR: Found {} data entries for fit_bases {}-{}-{}".format(
              np.sum(ind), *lmk))
          sys.exit(0)

        temp_LMK.append(lmk)
        temp_data_LMK.append(self.input_data_coeffs[ind,:])
        temp_data_var_LMK.append(self.input_data_coeffs_var[ind,:])
        #if self.C_distributions is not None:
        #  temp_C_distributions_LMK.append(self.C_distributions[ind])

      self.data_LMK = np.array(temp_LMK)
      self.data_coeffs = np.concatenate(temp_data_LMK, axis=0)
      self.data_coeffs_var = np.concatenate(temp_data_var_LMK, axis=0)
      #if self.C_distributions is not None:
      #  temp_C_distributions_LMK = np.concatenate(temp_C_distributions_LMK)
    
   
    # Prune dom axis
    self.dom = self.dom[self.dom_mask]
    self.calc_dom = np.expand_dims(self.dom, axis=0)
    self.data_coeffs = self.data_coeffs[:,self.dom_mask]
    self.data_coeffs_var = self.data_coeffs_var[:,self.dom_mask]
    #if self.C_distributions is not None:
    #  self.C_distributions = temp_C_distributions_LMK[:,self.dom_mask,:]
    self.atm_scat = self.atm_scat[self.dom_mask]
    for atm in self.scat_amps.keys():
      self.scat_amps[atm] = self.scat_amps[atm][self.dom_mask] 
    self.dist_sms_scat_amps = self.dist_sms_scat_amps[:,:,self.dom_mask]
    if not self.data_or_sim:
      if self.experimental_var is not None:
        self.experimental_var = self.experimental_var[:,self.dom_mask]
   

  def make_wiener_weight(self):
    do_wiener = True
    if "wiener" in self.data_params:
      do_wiener = self.data_params["wiener"]

    if do_wiener:
      # Wiener Filter
      if self.data_coeffs_var is not None:
        all_dists = calc_dists(self.atom_positions)
        dists = all_dists[self.dist_inds]
        C = np.complex(0,1)**self.data_Lcalc*8*np.pi**2/(2*self.data_Lcalc + 1)\
            *np.sqrt(4*np.pi*(2*self.data_Lcalc + 1))
        J_scale = 1./np.expand_dims(
            self.dom*np.expand_dims(dists[:,0], axis=-1), 0)
        Y = sp.special.sph_harm(-1*self.data_Kcalc, self.data_Lcalc,
            np.expand_dims(np.expand_dims(dists[:,2], axis=0), axis=-1),
            np.expand_dims(np.expand_dims(dists[:,1], axis=0), axis=-1))
        S = self.I*np.sum(np.real(self.dist_sms_scat_amps*C*Y*J_scale), axis=1)
        S = S**2


        N = copy(self.data_coeffs_var)
        if not self.data_params["isMS"]:
          N /= self.atm_scat**2

        self.wiener = S/(S+N)
      else:
        raise ValueError(
            "data_coeffs_var must be specified to calculate wiener weight")
    else:
      self.wiener = np.ones((1,1))

    if self.plot_setup:
      for lg in range(len(self.data_LMK)):
        fig, ax = plt.subplots()
        handles = []
        handles.append(ax.errorbar(self.dom, self.data_coeffs[lg,:],\
            np.sqrt(self.data_coeffs_var[lg,:]),\
            label="legendre {}-{}-{}".format(*self.data_LMK[lg])))
        ax.plot(self.dom, self.data_coeffs[lg,:], '-k')
        if np.prod(self.wiener.shape) > 1:
          ax2 = ax.twinx()
          ax2.plot(self.dom, self.wiener[lg,:], color='gray')
          ax2.tick_params(axis='y', labelcolor='gray') 
          ax.set_xlabel(r'q $[\AA]$')
          ax.set_ylabel("C [arb]")
          ax2.set_ylabel("Data Filter")
        ax.legend(handles=handles)
        fig.savefig("./plots/{}/data_coeffs_lg-{}-{}-{}.png".format(
          self.data_params["molecule"], *self.data_LMK[lg]))
        plt.close()
        #if self.data_LMK[lg][0] == 2 and self.data_LMK[lg][-1] == 0:
        #  print(self.data_coeffs[lg,:])

    """
    # Must normalize filter to account for the removed bins
    print("WEIN SHAPE", self.wiener.shape)
    w_norm_ = np.mean(self.wiener, axis=-1, keepdims=True)
    w_norm = 1./copy(w_norm_)
    w_norm[w_norm_==0] = 0
    self.wiener *= w_norm
    """


  def evaluate_scattering_amplitudes(self):
    """
    This function fills the scat_amps dictionary with interpolated scattering
    amplitudes evaluated at the reciprocal space measurement points and builds
    the atomic scattering.

        Parameters
        ----------

        Returns
        -------
    """

    self.atm_scat = np.zeros_like(self.dom)
    self.scat_amps = {}
    for atm in self.atom_types:
      if atm not in self.scat_amps:
        self.scat_amps[atm] = self.scat_amps_interp[atm](self.dom)
      self.atm_scat += self.scat_amps[atm]**2


  def rotate_to_principalI(self, R):
    """
    Rotate the molecular cartesian coordinates in R to the molecular frame

        Parameters
        ----------

        Returns
        -------
    """

    # Center of Mass
    R -= np.sum(R*self.mass[:,0], -2, keepdims=True)/np.sum(self.mass)

    # Calculate principal moment of inertia vectors
    I_tensor = self.calc_I_tensor(R)
    Ip, I_axis = np.linalg.eigh(I_tensor)

    return np.matmul(R, I_axis)[:,np.array([1,2,0])]


  def rotate_to_principalI_ensemble(self, R):
    """
    Rotate an ensemble molecular cartesian coordinates in R to the
    molecular frame

        Parameters
        ----------

        Returns
        -------
    """

    # Center of Mass
    R -= np.sum(R*self.mass[:,0], axis=-2, keepdims=True)/np.sum(self.mass)

    # Calculate principal moment of inertia vectors
    I_tensor = self.calc_I_tensor_ensemble(R)
    Ip, I_axis = np.linalg.eigh(I_tensor)

    return np.matmul(R, I_axis)[:,:,:,np.array([1,2,0])]


  """
  def get_molecule_init_geo(self):
    if not os.path.exists(self.data_params["init_geo_xyz"]):
      print("Cannot find xyz file: " + self.data_params["init_geo_xyz"])
      sys.exit(1)

    self.atom_types      = []
    self.atom_positions  = []
    with open(self.data_params["init_geo_xyz"]) as file:
      for i,ln in enumerate(file):
        if i == 0:
          Natoms = int(ln)
        elif i > 1:
          vals = ln.split()
          print("\t" + vals)
          self.atom_types.append(vals[0])
          pos = [float(x) for x in vals[1:]]
          self.atom_positions.append(np.array([pos]))

    self.atom_positions = np.concatenate(self.atom_positions, axis=0)
  """


    

  def calculate_coeffs_single_scipy(self, molecules):
    """
    Calculate the C coefficients using Scipy functions for an array-like
    of molecular geometries in cartesian space

        Parameters
        ----------
        molecules : 3D np.array of type float [N,atoms,xyz]
            An array of molecular cartesian cordinates

        Returns
        -------
        calc_coeffs : 3D np.array of type float [N,lmk,q]
            The calculated C coefficients
    """

    # Rotate molecule into the MF (Principal axis of I)
    R = self.rotate_to_principalI(molecules)

    # Calculate pair-wise vectors
    all_dists = calc_dists(R)
    dists = all_dists[self.dist_inds]

    # Calculate diffraction response
    C = np.complex(0,1)**self.data_Lcalc*8*np.pi**2/(2*self.data_Lcalc + 1)\
        *np.sqrt(4*np.pi*(2*self.data_Lcalc + 1))
    J = sp.special.spherical_jn(self.data_Lcalc, 
        self.calc_dom*np.expand_dims(dists[:,0], axis=-1))
    Y = sp.special.sph_harm(-1*self.data_Kcalc, self.data_Lcalc,
        np.expand_dims(np.expand_dims(dists[:,2], axis=0), axis=-1),
        np.expand_dims(np.expand_dims(dists[:,1], axis=0), axis=-1))


    # Sum all pair-wise contributions
    calc_coeffs = np.sum(np.real(self.dist_sms_scat_amps*C*J*Y), axis=1)

    # Subtract mean and normalize 
    # TODO Fix subtract mean
    #calc_coeffs -= np.expand_dims(np.mean(calc_coeffs, axis=-1), -1)
    calc_coeffs *= self.I

    return calc_coeffs


  def calculate_coeffs_ensemble_scipy(self, R, w):
    """
    Calculate the C coefficients using Scipy functions for an array-like
    of ensembles of molecular geometries in cartesian space

        Parameters
        ----------
        R : 4D np.array of type float [N,ensemble,atoms,xyz]
            An array of ensembles of molecular cartesian cordinates
        w : 2D np.array of type float [N,ensemble]
            The weights of each geometry in the ensemble

        Returns
        -------
        calc_coeffs : 2D np.array of type float [N,lmk,q]
            The calculated C coefficients
    """

    # Rotate molecule into the MF (Principal axis of I)
    #tic = time.time()
    R = self.rotate_to_principalI_ensemble(R).transpose((2,3,0,1))
    #print("\trotate time:", time.time()-tic)

    # Calculate pair-wise vectors
    all_dists = calc_dists(R)
    #print("all dists", R)
    dists = all_dists[self.dist_inds]

    # Calculate diffraction response
    #ttic = time.time()
    #tic = time.time()
    C = np.complex(0,1)**self.data_Lcalc*8*np.pi**2/(2*self.data_Lcalc + 1)\
        *np.sqrt(4*np.pi*(2*self.data_Lcalc + 1))
    #print("\tC time:", C.shape, time.time()-tic)
    #tic = time.time()
    inp = np.expand_dims(np.expand_dims(self.calc_dom, -1), -1)\
        *np.expand_dims(dists[:,0], axis=1)
    J = self.spherical_j(inp)
    #print("\tJ time:", inp.shape, J.shape, time.time()-tic)
    #tic = time.time()
    Y = sp.special.sph_harm(
        -1*np.expand_dims(np.expand_dims(self.data_Kcalc, -1), -1),
        np.expand_dims(np.expand_dims(self.data_Lcalc, -1), -1),
        np.expand_dims(np.expand_dims(dists[:,2], axis=0), axis=2),
        np.expand_dims(np.expand_dims(dists[:,1], axis=0), axis=2))
    #print("\tY:", Y.shape, time.time()-tic)
    #print("\tCJY time:", time.time()-ttic)

    # Sum all pair-wise contributions
    #tic = time.time()
    calc_coeffs = np.sum(np.real(
        np.expand_dims(np.expand_dims(self.dist_sms_scat_amps, -1), -1)\
        *np.expand_dims(np.expand_dims(C, -1), -1)*J*Y), axis=1)
    #print("\tsum time:", time.time()-tic)

    #plt.hist(calc_coeffs[-1,2,:], bins=25, weights=w[:,0])
    #plt.savefig("testDist.png")
    #plt.close()
    # Subtract mean and normalize 
    #calc_coeffs -= np.expand_dims(np.mean(calc_coeffs[:,:], axis=1), 1)
    calc_coeffs *= self.I

    return calc_coeffs.transpose((2,3,0,1))


  def calculate_coeffs_ensemble_multiProc(self, ensemble, weights):
    """
    Split the C coefficient calculations among different processors based
    on the number specified in the runtime parameter

    multiprocessing : int
        The number of cores used to split the calculation

        Parameters
        ----------
        ensemble : 4D np.array of type float [N,ensemble,atoms,xyz]
            An array of ensembles of molecular geometries
        weights : 2D np.array of type float [N,ensemble]
            An array of weights for the probability each geometry would appear
            in the ensemble

        Returns
        -------
        calc_coeffs : 3D np.array of type float [N,lmk,q]
            The calculated C coefficients for each ensemble
    """

    mp_stride = int(np.ceil(
        ensemble.shape[0]/self.data_params["multiprocessing"]))
    jobs = []
    return_dict = self.mp_manager.dict()
    for i in range(self.data_params["multiprocessing"]):
      p = mp.Process(target=self.calculate_coeffs_ensemble_multiProc_helper,
          args=(i,
            ensemble[i*mp_stride:(i+1)*mp_stride],
            weights[i*mp_stride:(i+1)*mp_stride],
            return_dict))
      p.start()
      jobs.append(p)
      if (i+1)*mp_stride >= ensemble.shape[0]:
        break

    for p in jobs:
      p.join()

    calc_coeffs = []
    for pNum in np.arange(len(return_dict)):
      calc_coeffs.append(return_dict[pNum])
    calc_coeffs = np.concatenate(calc_coeffs)
    del return_dict

    return calc_coeffs


  def calculate_coeffs_ensemble_multiProc_helper(
      self, procNum, R, weights, return_dict):
    """
    A helper function to handle the return_dict when using multiprocessing

        Parameters
        ----------
        procNum : int
            The identifying process number
        R : 4D np.array of type float [N,ensemble,atoms,xyz]
            An array of ensembles of molecular geometries
        weights : 2D np.array of type float [N,ensemble]
            An array of weights for the probability each geometry would appear
            in the ensemble
        return_dict : multiprocessing.mp_manager.dict() instance
            A dictionary that has shared memory with all the other processors 
            to save the calculated C coefficients for each process

        Returns
        -------
    """
    
    return_dict[procNum] = self.calculate_coeffs_ensemble_cpp(R, weights)


  def calculate_coeffs_ensemble_cpp(self, R, weights, verbose=False):
    """
    Calculate the C coefficients using the C++ implementation of the
    Spherical Bessel functions in this package

        Parameters
        ----------
        R : 4D np.array of type float [N,ensemble,atoms,xyz]
            An array of ensembles of molecular geometries
        weights : 2D np.array of type float [N,ensemble]
            An array of weights for the probability each geometry would appear
            in the ensemble
        return_dict : multiprocessing.mp_manager.dict() instance
            A dictionary that has shared memory with all the other processors 
            to save the calculated C coefficients for each process
        verbose : bool
            Print the time it takes to evaluate various parts of the calcuation

        Returns
        -------
        calc_coeffs : 3D np.array of type float [N,lmk,q]
            The calculated C coefficients for each ensemble
    """

    # Rotate molecule into the MF (Principal axis of I)
    if verbose:
      tic = time.time()
    R = self.rotate_to_principalI_ensemble(R).transpose((2,3,0,1))
    if verbose:
      print("\trotate time:", time.time()-tic)

    # Calculate pair-wise vectors
    all_dists = calc_dists(R)
    #print("all dists", R)
    dists = all_dists[self.dist_inds]

    # Calculate diffraction response
    if verbose:
      ttic = time.time()
      tic = time.time()
    C = np.complex(0,1)**self.data_LMK[:,0]*8*np.pi**2\
        /(2*self.data_LMK[:,0] + 1)\
        *np.sqrt(4*np.pi*(2*self.data_LMK[:,0] + 1))
    if verbose:
      print("\tC time:", C.shape, time.time()-tic)
      tic = time.time()
    
    Y = sp.special.sph_harm(
        -1*np.expand_dims(self.data_Kcalc, -1),
        np.expand_dims(self.data_Lcalc, -1),
        np.expand_dims(dists[:,2], axis=0),
        np.expand_dims(dists[:,1], axis=0))
    if verbose:
      print("\tY:", Y.shape, time.time()-tic)
      tic = time.time()

    calc_coeffs = calc_coeffs_cpp_helper(
        np.expand_dims(np.expand_dims(self.calc_dom, -1), -1)\
          *np.expand_dims(dists[:,0], axis=1),
        self.data_LMK[:,0], C, Y,
        self.dist_sms_scat_amps, weights)

    if verbose:
      print("\tcpp time:", time.time()-tic)
      print("\tCJY time:", time.time()-ttic)

    # Subtract mean and normalize 
    #calc_coeffs -= np.expand_dims(np.mean(calc_coeffs[:,:], axis=1), 1)
    calc_coeffs *= self.I

    return calc_coeffs.transpose((2,0,1))
 

  """
  def calculate_log_prob_density(self, R, n=0):

    calc_coeffs = self.calculate_coeffs(R)
   
    prob = np.mean(-0.5*(self.data_coeffs - calc_coeffs)**2\
        /self.data_coeffs_var)
    #    + np.log(1/np.sqrt(self.data_coeffs_var)))
   
    return prob
  """
  
  def default_log_prior(self, theta):
    """
    Returns a log prior of 0 indicating that all theta are equally likely to be selected

        Parameters
        ----------
        theta : 2D np.array of type float [walkers,thetas]
            Set of theta parameters to evaluate

        Returns
        -------
        zeros: 2D np.array of type float [walkers]
            An array of 0s
    """

    return np.zeros(theta.shape[0])


  def gaussian_log_likelihood(self, calc_coeffs):
    """
    Calculates the log likelihood of the calculated C coefficients with the
    given data assuming the error distribution of each C coefficient is 
    Gaussian distributed

        Parameters
        ----------
        calc_coeffs : 3D np.array of type float [N,lmk,q]
            The calculated C coefficients for each ensemble
        
        Returns
        -------
    """

    return -0.5*(self.data_coeffs - calc_coeffs)**2\
        /self.data_coeffs_var


  def log_likelihood_density(self, theta, n=0):

    # Convert parameters to cartesian coordinates
    molecules = self.theta_to_cartesian(theta)

    calc_coeffs = self.calculate_coeffs_ensemble(molecules, 1)

    prob = np.nanmean(np.nanmean(
          self.wiener*self.C_log_likelihood(calc_coeffs),
        axis=-1), axis=-1)
    #    + np.log(1/np.sqrt(self.data_coeffs_var)))
 
    return prob 


  """
  def log_likelihood_optimal(self, theta, n=0):

    # Convert parameters to cartesian coordinates
    molecules = self.theta_to_cartesian(theta)

    calc_coeffs = self.calculate_coeffs_ensemble(molecules)

    prob = np.nansum(np.nansum(
          self.wiener*self.C_log_likelihood(calc_coeffs),
        axis=-1), axis=-1)
    #    + np.log(1/np.sqrt(self.data_coeffs_var)))
 
    return prob
  """


  def log_likelihood_optimal(self, theta, n=0):
    print("IN OPT")

    # Simulate the molecular ensemble
    ensemble = self.density_generator(theta)   # Cartesian
    if len(ensemble) == 2:
      ensemble, weights = ensemble
    else:
      weights = np.ones((ensemble.shape[0], 1))


    tic = time.time()
    calc_coeffs = self.calculate_coeffs(ensemble, weights)
    tic = time.time()
    """
    calc_coeffs = np.zeros(
        (theta.shape[0], self.data_LMK[:,0].shape[0], self.dom.shape[0]))
    ind, ind_step = 0, int(np.ceil(100000./ensemble.shape[0]))
    print("calc time:", time.time()-tic)
    print("TEST CPP", calc_coeffs[450:455,:,100:150])
    calc_coeffs = np.zeros(
        (theta.shape[0], self.data_LMK[:,0].shape[0], self.dom.shape[0]))
    while ind < ensemble.shape[1]:
      ind_end = np.min([ensemble.shape[1], ind + ind_step])
      calc_coeffs_ = self.calculate_coeffs_ensemble(ensemble[:,ind:ind_end], 1)
      calc_coeffs = calc_coeffs\
          + np.sum(calc_coeffs_\
            *np.expand_dims(np.expand_dims(weights[:,ind:ind_end], -1), -1), 1)
      #print("temp", ind, ind_step, ensemble[:,ind:ind_end].shape, calc_coeffs_.shape, calc_coeffs.shape)
      ind = ind_end
    print("TEST PY", calc_coeffs[450:455,:,100:150])

    print("calc time:", time.time()-tic)
    """
  
    prob = np.nansum(np.nansum(
          self.wiener*self.C_log_likelihood(calc_coeffs),
        axis=-1), axis=-1)
    #    + np.log(1/np.sqrt(self.data_coeffs_var)))
 
    return prob

  def log_likelihood(self, theta):
    """
    Calculate the log likelihood of with the theta parameters given the
    observed or simulated C coefficients by building an ensemble from the
    density generator

        Parameters
        ----------
        theta : 2D np.array of type float [walkers,thetas]
            An array of theta (model) parameters

        Returns
        -------
        prob : 1D np.array of type float [walkders]
            The log likelihood for each set of theta parameters
    """

    # Simulate the molecular ensemble
    ensemble = self.density_generator(theta)   # Cartesian
    
    if len(ensemble) == 2:
      ensemble, weights = ensemble
    else:
      weights = np.ones((ensemble.shape[0], 1))

    calc_coeffs = self.calculate_coeffs(ensemble, weights)

    prob = np.nansum(np.nansum(
          self.wiener*self.C_log_likelihood(calc_coeffs),
        axis=-1), axis=-1)
    #    + np.log(1/np.sqrt(self.data_coeffs_var)))

    return prob


  def log_probability(self, theta):
    """
    Calculates and combines the log likelihood and the log prior for each
    set of theta (model) parameters later used to compare between theta
    parameters in the MCMC

        Parameters
        ----------
        theta : 2D np.array of type float [walkers,thetas]
            An array of theta (model) parameters

        Returns
        -------
        prob : 1D np.array of type float [walkders]
            The log likelihood fnd the log prior
    """

    # Evaluate log prior
    lprior = self.log_prior(theta)

    # Evaluate log likelihood
    llike = self.log_likelihood(theta)
    #print("## Backend log prob ##")
    #print(self.sampler.backend.log_prob[-1,:])
    #print("## Chain ##")
    #print(self.sampler.backend.chain[-1])

    return lprior + llike


  def setup_sampler(self, nwalkers=None, ndim=None, expect_file=False):
    """
    This function sets up the MCMC sampler used for the Metropolis Hastings
    Algorithm. It firsts imports previously saved backends and returns the 
    sampler to the previous state. If there is no previously saved file it 
    creates a new backend and sampler.

        Parameters
        ----------
        nwalkers : int
            The number of independent walkers (chains) to evaluate in the MCMC
        ndim : int
            The number of theta (model) parameters
        expect_file : bool
            If False (default) then one does not expect a previosly saved file,
            otherwise if True and it cannot find a previosly saved file it will
            raise a RunTimeError
            
        Returns
        -------
        exists : bool
            True if the backend and state were loaded from a previosly saved file
        last_walker_pos : 2D np.array of type float [walkers,thetas]
            The last set of theta parameters of the loaded file, or None if there
            wasn't a file to load
    """

    fileName = os.path.join(self.data_params["output_dir"], self.get_fileName() + ".h5")

    exists = os.path.exists(fileName)
    if (not exists and expect_file):
      raise RuntimeError(
          "Expected file {} cannot be found!!!".format(fileName))

    if not exists and (nwalkers is None or ndim is None):
      raise RuntimeError(
          "xpected file {} cannot be found and nwalkers or ndim is None!!!".format(fileName))

    last_walker_pos = None
    if exists:
      backend, last_walker_pos = self.load_emcee_backend(fileName)
      nwalkers, ndim = last_walker_pos.shape
    else:
      print("INFO: {} does not exist, creating new backend".format(fileName))
      backend = emcee.backends.Backend()
   
    print("INFO: Setting up MCMC")
    self.sampler = emcee.EnsembleSampler(
        nwalkers, ndim, self.log_probability,
        backend=backend, vectorize=True)

    return exists, last_walker_pos


  def run_mcmc(self, walker_init_pos, Nsteps):
    """
    This function runs the MCMC that evaluates the Metropolis Hastings
    Algorithm. It sets up the sampler, runs it in batches defined by a
    runtime parameter, and saves the results after each batch. It will 
    run for at least 100 autocorrelation times and once the autocorrelation
    time change is less than 1% have converged or the minimum number of 
    iterations has been set by the runtime parameter 'min_acTime_steps'.

    Nwalkers : int
        The number of MCMC walkers (chains)
    run_limit : int
        The number of MCMC steps within a single batch
    min_acTime_steps : int
        The minimum number of autocorrelation times until the MCMC can
        converge

        Parameters
        ----------
        walker_init_pos : 2D np.array of type float [walkers,thetas]
            Either the starting points for each walker (chain) or the last
            theta parameters from a previosly saved file
        Nsteps : int 
            The number of sampling steps in each batch: "run_limit"

        Returns
        -------
    """

    nwalkers, ndim = walker_init_pos.shape
    _, last_walker_pos = self.setup_sampler(nwalkers, ndim)

    print("INFO: Running MCMC")
    if last_walker_pos is not None:
      walker_init_pos = last_walker_pos

    if self.tau_convergence is None:
      self.tau_convergence = [np.ones(ndim)*np.inf]
    sample_limit = True
    if "min_acTime_steps" in self.data_params:
      tau = self.sampler.get_autocorr_time(tol=0)
      sample_limit = np.all(tau*self.data_params["min_acTime_steps"]\
          < self.sampler.iteration)

    while not self.has_converged or not sample_limit:
      # Run MCMC
      self.sampler.run_mcmc(walker_init_pos, Nsteps, progress=True)

      # Calculate mean autocorrelation time
      tau = self.sampler.get_autocorr_time(tol=0)
      self.tau_convergence.append(copy(tau))
      autocorr = np.mean(tau)

      """
      print("will plot", not np.isnan(autocorr), tau) 
      s = self.sampler.get_chain(discard=90,
              thin=int(30))
      print(s.shape)
      print(np.mean(np.reshape(s, (-1, 3)), 0))
      print(np.std(np.reshape(s, (-1, 3)), 0))
      if s.shape[0] > 0:
        print(np.amax(s[:,:,2]))
      """
      if not np.isnan(autocorr):
        # Plot Progress
        if "plot_progress" in self.data_params:
          if self.data_params["plot_progress"]:
            self.plot_emcee_results(
              self.sampler.get_chain(discard=3*int(autocorr),
                  thin=int(autocorr)))

        # Check convergence
        conv1 = np.all(tau*100 < self.sampler.iteration)
        #conv1 = conv1 + (np.all(tau*20 < self.sampler.iteration) and np.max(tau) > 250)
        conv2 = np.all(np.abs(self.tau_convergence[-2] - tau)/tau < 0.01)
        if "min_acTime_steps" in self.data_params:
          sample_limit = np.all(tau*self.data_params["min_acTime_steps"]\
              < self.sampler.iteration)

        self.has_converged = conv1*conv2 or self.has_converged
        
        print("\tSample {}: mean tau = {} / convergence {} {}".format(
            self.sampler.iteration, tau, conv1, conv2))
      else:
        self.has_converged = False
      
      # Save MCMC
      self.save_emcee_backend()

      # End Jobs that take too long
      if np.amax(tau) > 500 and self.sampler.iteration/np.amax(tau) > 15:
        break

      # Start new mcmc run at the end of the previous
      walker_init_pos = self.sampler.backend.chain[-1]

    fileName = self.get_fileName()
    print("INFO: Finished with {}".format(fileName[fileName.rfind("/"):]))


  def load_emcee_backend(self, fileName):
    """
    Load the state and results (backend) of a previously saved emcee Sampler
    instances and print if the chain has converged or not

        Parameters
        ----------
        fileName : string
            Address of the h5 file that contains the backend

        Returns
        -------
        backend : emcee.backends.Backends instance
            The emcee.Sampler backend
        theta_last : 2D np.array of type float [walkers,thetas]
            The last matrix of theta parameters saved
    """

    print("INFO: Loading backend {}".format(fileName))
    backend = emcee.backends.Backend()
    with h5py.File(fileName, "r") as h5:
      #if h5["nwalkers"][...] != nwalkers or h5["ndim"][...] != ndim:
      #  print("ERROR: nwalkers or ndims do not match: {} != {} / {} != {}".format(
      #      h5["nwalkers"][...], nwalkers, h5["ndim"][...], ndim))
      #  sys.exit(0)
      backend.nwalkers = int(h5["nwalkers"][...])
      backend.ndim = int(h5["ndim"][...])
      backend.chain = h5["chain"][:]
      backend.accepted = h5["accepted"][:]
      backend.log_prob = h5["log_prob"][:]
      backend.iteration = len(backend.chain)
      backend.random_state = None
      backend.blobs = None
      backend.initialized = True
      if "blobs" in h5.keys():
        backend.blobs = h5["blobs"][:]

      self.has_converged = h5["has_converged"][...]
      self.tau_convergence = h5["tau_convergence"][:].tolist()

    if self.has_converged:
      print("\t The chain has converged")
    else:
      print("\t The chain has NOT converged")

    return backend, backend.chain[-1]


  def save_emcee_backend(self):
    """
    Saves the state and results (backend) of the current emcee Sampler

        Parameters
        ----------

        Returns
        -------
    """
   
    fileName = os.path.join(
        self.data_params["output_dir"], self.get_fileName() + ".h5")
    print("Saving {}".format(fileName))
    with h5py.File(fileName, "w") as h5:

      ###  Saving Backend  ###
      h5.create_dataset("chain", data=self.sampler.get_chain())
      h5.create_dataset("nwalkers", data=self.sampler.backend.nwalkers)
      h5.create_dataset("ndim", data=self.sampler.backend.ndim)
      h5.create_dataset("accepted", data=self.sampler.backend.accepted)
      h5.create_dataset("log_prob", data=self.sampler.backend.log_prob)
      if self.sampler.backend.blobs is not None:
        h5.create_dataset("blobs", data=self.sampler.backend.blobs)

      ###  Saving Convergernce Parameters  ###
      h5.create_dataset("has_converged", data=self.has_converged)
      if np.all(self.tau_convergence[0] == np.inf):
        h5.create_dataset("tau_convergence", data=self.tau_convergence[1:])
      else:
        h5.create_dataset("tau_convergence", data=self.tau_convergence)

      ###  Saving Results  ###
      try:
        tau = self.sampler.get_autocorr_time()
      except Exception as e:
        print(str(e))
        tau = self.sampler.get_autocorr_time(tol=0)
      h5.create_dataset("autocorr_times", data=tau)

      if np.all(np.isnan(tau)): 
        tau = np.ones_like(tau)

      tau_ = copy(tau)
      tau_[np.isnan(tau)] = 1
      max_ac = int(np.amax(tau_))
      if self.sampler.backend.iteration > 4*max_ac:
        h5.create_dataset("filtered_chain",
            data=self.sampler.get_chain(discard=3*max_ac, thin=max_ac))
      else:
        h5.create_dataset("filtered_chain", data=np.array([False]))

  
  def get_mcmc_results(self,
      log_prob=False, labels=None, plot=True, thin=True):
    """
    Load the MCMC results into the class and the previous state, as well
    as plot the correlations between theta parameters and the chain history

        Parameters
        ----------
        log_prob : bool
            If True return the log likelihood of each theta parameter
        labels : list of strings
            A list of labels for each theta (model) parameter, if None then
            do not use labels in the plots
        plot : bool
            If True plots the correlations between theta (model) parameters in
            addition to the the chains' history
        thin : True
            If True it thins the chains by the autocorrelaiton times so each 
            entrance is uncorrelated with the previous one

        Returns
        -------
        chain : 2D np.array of type float [N,thetas]
            The flattened chains from the MCMC that make up the marginalized
            posterior
        lp : 2D np.arrya of type float [N]
            The log lkelhood probabilities of each theta parameter if log_prob
            is True, otherwise this is not returned
    """

    # Import MCMC results if not already available
    if self.sampler is None:
      try:
        self.setup_sampler(expect_file=True)
      except RuntimeError:
        print("INFO: Results do not exist.")
        return False

    #####  Get Autocorrelation Times For Each Theta  #####
    try:
      tau = self.sampler.get_autocorr_time()
    except Exception as e:
      print(str(e))
      tau = self.sampler.get_autocorr_time(tol=0)

    print("INFO: Autocorrelation Times", tau)
    max_ac = np.amax(tau).astype(int)

    if thin:
      chain = self.sampler.get_chain(
          discard=3*max_ac, thin=max_ac)
      if log_prob:
        lp = self.sampler.get_log_prob(
            discard=3*max_ac, thin=max_ac)
    else:
      chain = self.sampler.get_chain(discard=3*max_ac)
      if log_prob:
        lp = self.sampler.get_log_prob(discard=3*max_ac)


    if plot:
      self.plot_emcee_results(chain, labels=labels)

    if log_prob:
      return np.reshape(chain, (-1, chain.shape[-1])), np.reshape(lp, (-1))
    else:
      return np.reshape(chain, (-1, chain.shape[-1]))



  def plot_emcee_results(self, samples, labels=None):
    """
    Plots the the trajectory of each walker (chain) in the sampler, as well
    as the 2d projections of each pair of theta (model) parameters.

        Parameters
        ----------
        samples : 3D np.array of type float
            The history of all the walkers (chains) to be plotted
        labels : list of strings
            A list of strings to label each theta parameter, if None then uses
            generic labels

        Returns
        -------
    """

    if samples is None:
      print("WARNING: Cannot plot with samples = None, Skipping!!!")
      return

    fileName_base = os.path.join("./plots", self.get_fileName())
    #####  Plot Trajectory in Theta of Each Walker  #####
    #samples = self.sampler.get_chain()
    Nsamples = samples.shape[-1]
    if labels is None or "labels" not in self.data_params:
      labels = []
      for i in range(Nsamples):
        labels.append("r$\theta_{}$".format(i))
    else:
      labels = self.data_params["labels"]

    fig, axs = plt.subplots(Nsamples, figsize=(2.5*Nsamples, 10), sharex=True)
    for i in range(len(labels)):
      axs[i].plot(samples[:, :, i], "k", alpha=0.3)
      axs[i].set_xlim(0, len(samples))
      axs[i].set_ylabel(labels[i])
      axs[i].yaxis.set_label_coords(-0.1, 0.5)

    axs[-1].set_xlabel("step number");
    plt.tight_layout()
    fig.savefig(fileName_base + "_theta_trajectories.png")
    plt.close()

    #####  Make Corner Plot  #####
    #fig = plt.figure()
    fig = corner.corner(
        np.reshape(samples, (-1, samples.shape[-1])), labels=labels)
    fig.savefig(fileName_base + "_correlation_plots.png")
    plt.close()


  def calculate_I0(self, skip_scale=False, plot=True):
    """
    Fits the input C200 contribution to the C200 calculation or uses the
    runtime parameter and plots the input and scaled calculation

    I0_scale : float
        If None then fit C200, else use the given value

        Parameters
        ----------
        skip_scale : bool
            If True do not fit or change I, else do so (default)
        plot : bool
            If True (default) plot the given C coefficients and the scaled calculated C coefficients

        Returns
        -------
    """

    print("INFO: Initial thetas used to fit I0")
    print("\t", self.data_params["init_thetas"])
    ensemble = self.density_generator(
        np.expand_dims(np.array(self.data_params["init_thetas"]), 0))
    if len(ensemble) == 2:
      ensemble, weights = ensemble
    else:
      weights = np.ones((ensemble.shape[0], 1))

    calc_coeffs = self.calculate_coeffs(ensemble, weights)

    # Calculate Scale Factor
    if not skip_scale and self.data_or_sim:
      self.I = np.ones((1,1))
      if "I_scale" in self.data_params:
        if self.data_params["I_scale"] is None:
          self.fit_I0(calc_coeffs)
        else:
          self.I = self.data_params["I_scale"]
      else:
        #TODO FIX THIS OPTION, fig_I0 does not exist
        self.fig_I0(calc_coeffs)

    # Plot initial theta coeffs vs data
    if plot and self.plot_setup:
      plot_coeffs = self.I*calc_coeffs[0]
      for i in range(plot_coeffs.shape[0]):
        plt.errorbar(self.dom, self.data_coeffs[i,:], np.sqrt(self.data_coeffs_var[i,:]))
        plt.plot(self.dom, plot_coeffs[i,:], '-k')
        fileName = os.path.join("plots", self.get_fileName())
        plt.savefig(fileName + "_scaleInit-{}-{}-{}.png".format(*self.data_LMK[i,:]))
        plt.close()


  def fit_I0(self, calc_coeffs, data=None, var=None, return_vals=False):
    """
    Fit the C200 coefficient for the intensity of the diffraction probe (I0)
    by minimizing the chi square

        Parameters
        ----------
        calc_coeffs : 3D np.array of type float [1,lmk,q]
            The calculated C coefficients
        data : 2D np.array of type float [lmk,q] or None
            The data to fit the I0 coefficient to, will use the imported or
            simulated data if None
        var : 2D np.array of type float [lmk,q] or None
            The variance of the data to fit the I0 coefficient to, will 
            use the imported or simulated data if None
        return_vals : bool
            Will return the fit I0 coefficient and its standard deviation
            if set to True. False by default.

        Returns
        -------
        I : float
            The fit I0 coefficient 
        I_std : float
            The standard deviation of the fit I0 coefficient
    """
    
    print("WARNING: Setting I0 based upon the same C coefficients used to access the molecular frame. Highly discouraged")
    if data is None:
      data = self.data_coeffs
    if var is None:
      var = self.data_coeffs_var
    I = np.nansum(calc_coeffs/var*data, -1)\
        /np.nansum(calc_coeffs/var*calc_coeffs, -1)
    I_std = np.sqrt(1./np.sum(calc_coeffs**2/var, -1))
    I = I[0]
    if len(np.array(I).shape) < 2:
      I = np.reshape(I, (1, 1))

    if return_vals:
      return I, I_std
    else:
      self.I = I
      self.I_std = I_std


  def remove_global_offset(self):
    
    if self.ensemble_generator is not None:
      # Get ensemble of molecules
      ensemble = self.ensemble_generator(
          np.expand_dims(np.array(self.data_params["sim_thetas"])[:-1], 0),
          N=int(self.data_params["sim_thetas"][-1]))
      if len(ensemble) == 2:
        ensemble, weights = ensemble
      else:
        weights = np.ones((ensemble.shape[0], 1))

      calc_coeffs = self.calculate_coeffs(ensemble, weights)[0]
      print("AAAA", self.calculate_coeffs(ensemble, weights).shape)
    else:
      raise ValueError(
          "Must specify the ensemble generator and parameter 'sim_thetas'")


    if "scale" not in self.data_params:
      I_init, I_std_init = fit_I0(calc_coeffs, return_vals=True,
          data=self.data_coeffs, var=self.data_coeffs_var)
    elif self.data_params["scale"] is None: 
      I_init, I_std_init = fit_I0(calc_coeffs, return_vals=True,
          data=self.data_coeffs, var=self.data_coeffs_var)
    else:
      I_init, I_std_init = np.array([[self.data_params["scale"]]]), 0

    figs, axs0, axs1 = [], [], []
    for il in range(self.data_LMK.shape[0]):
      fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2.5, 1]})
      figs.append(fig)
      axs0.append(axs[0])
      axs1.append(axs[1])
    plot_folder = os.path.join("plots", self.data_params["molecule"])

    # Plot the fourier power spectrum and filter
    b, a = sp.signal.butter(*self.data_params["global_offset"],
        btype="lp", analog=True)
    w, h = sp.signal.freqs(b, a)
    self.plot_filter(self.data_coeffs, (w, h), plot_folder, axs=axs1)
    
    # Low pass filter the data
    nan_mask = np.isnan(self.data_coeffs) + np.isnan(self.data_coeffs_var)
    filt_inp = copy(self.data_coeffs[nan_mask])
    print("TEST SHAPES", self.data_coeffs.shape, nan_mask.shape, filt_inp.shape)
    i_ = np.arange(filt_inp.shape[0]).astype(int)
    inds = np.zeros(filt_inp.shape[0]).astype(int)
    while np.any(np.isnan(filt_inp[i_,inds])):
      ii = np.isnan(filt_inp[i_,inds])
      inds[ii] += 1
    for i in range(filt_inp.shape[0]):
      filt_inp[i,:inds[i]] = filt_inp[i,inds[i]+1]
    sos = sp.signal.butter(*self.data_params["global_offset"],
        btype="lp", output='sos')
    smoothed_data = sp.signal.sosfiltfilt(sos, filt_inp)

    print("NANS smooth data", np.sum(np.isnan(smoothed_data)))
    # Fit data+offset to calculated coefficients
    fit_data = np.concatenate([np.ones(smoothed_data.shape + (1,)),
        np.expand_dims(smoothed_data, -1)], -1)
    print("FIT D SHAPE", fit_data.shape, self.dom.shape)
    print("inp nans", np.sum(np.isnan(fit_data)), np.sum(np.isnan(self.data_coeffs_var)))
    norm = np.linalg.inv(np.einsum('aib,aic->abc', fit_data,
        fit_data/np.expand_dims(self.data_coeffs_var, -1)))
    fit = np.einsum('abi,ai->ab', norm, 
        np.sum(fit_data*np.expand_dims(calc_coeffs, -1)\
          /np.expand_dims(self.data_coeffs_var,-1),
          -2))
    fit_std = np.sqrt(
        norm[:,np.arange(norm.shape[-1]), np.arange(norm.shape[-1])]) 
    print("FIT", fit, fit_std)
    print("WARNING: Calculating offset and I (scale) from the first LMK entry only, all other data points are ignored!")
    self.I, self.I_std = 1./fit[0,1], 1./fit_std[0,1]
    self.offset, self.offset_std = fit[:,0]/fit[:,1], fit_std[:,0]/fit[:,1]
    print("FIT RES", self.I, self.I_std, self.offset, self.offset_std)

    # Adding offset and plotting
    base_fileName = self.get_fileName(suffix="global_offset")
    for il in range(self.data_LMK.shape[0]):
      axs0[il].plot(self.dom, self.data_coeffs[il]/I_init[0,0],
          color='gray', alpha=0.7, label="Data") 
      axs0[il].plot(self.dom, smoothed_data[il]/I_init[0,0],
          color='gray', ls='--', label="Smoothed Data")
      axs0[il].plot(self.dom, calc_coeffs[il], color='k', label="Simulation")
    self.data_coeffs += self.offset
    for il in range(self.data_LMK.shape[0]):
      axs0[il].plot(self.dom, (smoothed_data+self.offset)[il]/self.I,
          color='k', ls='--', label="Fit to Simulation")
      axs0[il].set_xlim(self.dom[0], self.dom[-1])
      axs0[il].set_xlabel(r'q $[\AA]$')
      axs0[il].set_ylabel('C [arb]')
      fig.legend()
      plt.tight_layout()
      figs[il].savefig(os.path.join(plot_folder,
          "global_offset_lmk-{}-{}-{}.png".format(*self.data_LMK[il])))


  def get_fileName(self, folder_only=False, suffix=None):
    """
    Create the file namd and/or address that uniquely defines the data or
    simulation based on the given runtime parameters that define this instance

        Parameters
        ----------
        folder_only : bool
            If True only returns the address of the folder
        suffix : string
            If not None then add this suffix to the end of the file name

        Returns
        -------
        name : string
            Either the only the folder or the entire address of the file
            depending on folder_only
    """

    dtype = "data"
    if "simulate_data" in self.data_params or not self.data_or_sim:
      if self.data_params["simulate_data"]:
        dtype = "sim"

    lg_name = "lg"
    for l in np.sort(np.unique(self.data_params["fit_bases"][:,0])):
      lg_name += "-{}".format(int(l))

    if "density_model" in self.data_params:
      dens_type = self.data_params["density_model"]
    else:
      dens_type = ""

    adm_type = ""
    if "ADM_params" in self.data_params:
      args = self.data_params["ADM_params"]
      if "temperature" in args:
        adm_type += str(args["temperature"]) + "K_"
      if "intensity" in args:
        adm_type += str(args["intensity"]) + "TW_"
      if "probe_FWHM" in args:
        adm_type += str(args["probe_FWHM"]) + "fs"

      if len(adm_type) > 0:
        if adm_type[-1] == "_":
          adm_type = adm_type[:-1]

    folder = os.path.join(self.data_params["molecule"],
        self.data_params["experiment"], dtype, dens_type, adm_type)
    if folder_only:
      return folder

    fileName = "results_{}_range-{}-{}".format(
        dtype, 
        float(self.data_params["fit_range"][0]),
        float(self.data_params["fit_range"][1]))

    if "simulate_data" in self.data_params:
      if self.data_params["simulate_data"]:
        etype = "data"

        if "simulate_error" in self.data_params:
          etype, variance_scale = self.data_params["simulate_error"]
          if etype == "StoN" or etype == "constant_sigma" or etype == "constant_background":
            if etype == "StoN":
              variance_scale = variance_scale[0]
            etype = etype.lower()

            fileName += "_error-{0}_scale-{1:.3g}".format(
                etype, variance_scale)
          elif etype == "data":
            fileName += "_error-{0}_filter-{1}-{2:.3g}".format(
                etype, *variance_scale)

    if suffix is not None:
      fileName += suffix
    return os.path.join(folder, fileName)


  def save_results(self, probs, distributions):

    fileName = os.path.join(self.data_params["output_dir"],
        self.get_fileName(np.prod(probs.shape[1])))
    fileName += ".h5"

    # Save results in h5 format
    with h5py.File(fileName, "w") as h5:
      h5.create_dataset("log_probabilities", data=probs)
      h5.create_dataset("geometries", data=distributions)
      h5.create_dataset("order", data=np.arange(len(probs)))
      h5.create_dataset("I0", data=self.I)
      h5.create_dataset("perturb_range", data=self.data_params["perturb_range"])
      h5.create_dataset("multiprocessing", data=self.data_params["multiprocessing"])



  def get_results(self):
    
    fileName = os.path.join(
        self.data_params["output_dir"], self.get_fileName(1000))
    ind = fileName.find("_N-") + 3
    files = glob.glob(fileName[:ind] + "*.h5")
    N = 0
    for fl in files:
      n = int(fl[ind:-3])
      if n > N:
        N = n
   
    if len(files) > 0:
      fileName = fileName[:ind] + str(N) + ".h5"
     
      print(fileName)
      with h5py.File(fileName, "r") as h5:
        probs       = h5["log_probabilities"][:]
        geometries  = h5["geometries"][:]
        order       = h5["order"][:]
        self.I      = h5["I0"][...]
        self.data_params["scale"] = h5["I0"][...]
        self.data_params["perturb_range"] = h5["perturb_range"][:]
        if self.do_multiprocessing:
          if "multiprocessing" not in h5.keys():
            mp = h5["Nnodes"][...]
          else:
            mp = h5["multiprocessing"][...]
          if self.data_params["multiprocessing"] != mp:
            raise RuntimeError(
                "The provided number of nodes 'multiprocessing' does not match"\
                "the {} nodes in {}".format(mp, fileName))
        self.data_params["multiprocessing"] = mp

      print("INPUT", probs.shape, geometries.shape)

      """
      os.path.join(self.data_params["output_dir"], self.data_params["molecule"],
          "log_probabilities-{}.npy".format(N)) 
      with open(fName, "rb") as file:
        probs = np.load(file)

      fName = os.path.join(, self.data_params["molecule"],
          "molecules-{}.npy".format(N)) 
      with open(fName, "rb") as file:
        distributions = np.load(file)

      fName = os.path.join(self.data_params["output_dir"], self.data_params["molecule"],
          "order-{}.npy".format(N)) 
      with open(fName, "rb") as file:
        order = np.load(file)


      probs = probs[order].tolist()
      distributions = [np.expand_dims(x, axis=0) for x in distributions[order]]
      """

      probs = [list(p) for p in probs]
      geometries = [list(g) for g in geometries]
      print("adsfasdf", len(geometries), len(geometries[0]), geometries[0][0].shape)
      return probs, geometries

    else:
      return\
          [[] for i in range(self.data_params["multiprocessing"])],\
          [[] for i in range(self.data_params["multiprocessing"])]


  def setup_calculations(self):
    """
    This function sets up the spherical bessel function implementation used
    to calculate the C coefficients bases on the value of the runtime 
    parameter "calc_type"

    0 <- C++ implementation (Recommended, but cannot include very low q)
    1 <- Scipy implementation (Slowest but correct for all q values)
    2 <- Optimized Python implementation (Slower than 0 with the same errors)

    If using option 0 or 2 one MUST check the "Checking Scale of Spherical
    Bessel Function Error" output and the check_jl{l}_calculations
    plots to make sure the residual is negligable for the data or 
    simulation's reciprocal space range.

        Parameters
        ----------

        Returns
        -------
    """

    # By default use scipy implementation
    if "calc_type" not in self.data_params:
      self.data_params["calc_type"] = 1

    if self.data_params["calc_type"] == 0 or self.do_multiprocessing:
      # Use the C++ implementation of spherical bessel functions
      
      if np.sum(self.data_LMK[:,0] % 2) == 0:
        ind = 0
        lmk = self.data_LMK[:,0]
        keep_inds = np.ones(len(lmk)).astype(bool) 
        for i in np.arange(np.amax(lmk)//2+1):
          if i*2 not in lmk:
            for j in np.arange(len(lmk)):
              if i*2 < lmk[j]:
                lmk = np.insert(lmk, j, i*2, axis=0)
                keep_inds = np.insert(keep_inds, j, False, axis=0)
                break
     
        def calc_even_only(x):
          return spherical_j(x, lmk)[keep_inds]

        self.spherical_j = calc_even_only
      if self.do_multiprocessing:
        self.calculate_coeffs = self.calculate_coeffs_ensemble_multiProc
      else:
        self.calculate_coeffs = self.calculate_coeffs_ensemble_cpp

    elif self.data_params["calc_type"] == 1:
      # Using scipy implementation of spherical bessel functions
      def numpy_jn(x):
        return sp.special.spherical_jn(self.data_Lcalc, x) 

      self.spherical_j = numpy_jn
      self.calculate_coeffs = self.calculate_coeffs_ensemble_scipy


    elif self.data_params["calc_type"] == 2:
      # Using the below python implementation of spherical bessel functions
      
      if np.sum(self.data_LMK[:,0] % 2) == 0:
        N = (np.unique(n)/2).astype(int)
        N_max = int(np.amax(N))
        N_elements = 2*N_max
        counts = []
        for nn in np.unique(n):
          counts.append(np.sum(nn == n))

        # Indices of invx for sin (even) and cos (odd) terms
        even_inds = np.where(np.arange(N_elements+1)%2 == 0)[0]
        odd_inds = np.where(np.arange(N_elements+1)%2 == 1)[0]

        # Remove contributions that are not in given n
        keep_inds = np.ones(int(np.amax(N))+1).astype(bool)
        remove_inds = []
        for i in range(len(keep_inds)):
          if i not in N:
            keep_inds[i] = False
            remove_inds.append(i)

        # only nonzero coeffs: 1/x^n | n is even/odd for left/right
        # This is filled in by hand, use wolfram alpha to find coefficients
        scales = [
            [np.array([1]), np.array([0])],
            [np.array([-1, 3]), np.array([-3])],
            [np.array([1, -45, 105]), np.array([10, -105])],
            [np.array([-1, 210, -4725, 10395]), np.array([-21, 21*60, -21*495])],
            [np.array([1, -630, 51975, -945945, 2027025]),
                9*np.array([4, -770, 30030, -225225])]]

        #for i in reversed(remove_inds):
        #  del scales[i]

        def calc_even_only(x):
          res = []

          tt = time.time()
          invx = [1./x]
          invx.append(invx[0]/x)
          invx = np.array(invx)

          # Calculate j0
          tt = time.time()
          #res.append((np.sin(x)*invx[0]))
          #j0_ = np.cos(x)*invx[0]
          res.append(np.sin(x))
          j0_ = np.sqrt(1-res[0]**2)*invx[0]
          neg_inds = np.where((x%(2*np.pi)>np.pi/2) & (x%(2*np.pi)<3*np.pi/2))
          j0_[neg_inds] *= -1
          res[0] = res[0]*invx[0]

          # Calculate j2
          tt = time.time()
          #res.append(np.exp(np.log(scales[1][0][0] +  scales[1][0][1]*invx[1]) + np.log(res[0]))
          #    + np.exp(np.log(scales[1][1][0]*invx[0])+np.log(j0_)))
          res.append(((scales[1][0][0] +  scales[1][0][1]*invx[1])*res[0]
              + scales[1][1][0]*invx[0]*j0_).astype(np.double))

          tt = time.time()
          for i in np.arange(N_max+1):
            if i == 0 or i == 1:
              continue

            t = time.time()
            nn = (i-1)*2.
            #res.append((4.*nn*(nn+1)*invx[1] - (1 + (nn+1)/(nn-1)))*res[-1] - res[-2]*(nn+1)/(nn-1))
            res.append(
                ((2*nn+1)*(2*(nn+1)+1)*invx[1] - (1 + (2*(nn+1)+1)/(2*(nn-1)+1)))*res[-1]\
                - res[-2]*(2*(nn+1)+1)/(2*(nn-1)+1))

          tt = time.time()
          for i in np.arange(N_max+1):
            if x[0,0,0,0] == 0:
              if i == 0:
                res[-1][:,0,:,:] = 1.
              else:
                res[-1][:,0,:,:] = 0

          tt = time.time()
          ic = 0
          for i in range(len(res)):
            if keep_inds[i]:
              res[i] = np.tile(res[i], (counts[ic],1,1,1,1))
              ic += 1
          for i in reversed(np.where(np.invert(keep_inds))[0]):
            del res[i]

          tt = time.time()
          a = np.concatenate(res, 0)
          return np.concatenate(res, 0)

        self.spherical_j = calc_even_only
      self.calculate_coeffs = calculate_coeffs_ensemble_scipy
      """
      def calc_even_only(x):
          res = []
          j0 = np.sin(x)/x
          j0_ = np.cos(x)/x
          if keep_inds[0]:
            res.append(copy(j0))

          invx = [np.ones_like(x, dtype=np.double)]
          print("going to", N_elements)
          for i in range(N_elements):
            invx.append((invx[-1]/x).astype(np.double))
            #invx.append(-1*(i+1)*np.log(x))
          #invx = np.log(np.array(invx))
          invx = np.array(invx).astype(np.double)

          res = []
          for i,nn in enumerate(N):
            print("SHA", np.reshape(scales[i][0], (-1,1,1,1,1)).shape, invx[even_inds[:len(scales[i][0])]].shape, j0.shape)
            print("SHA1", np.reshape(scales[i][1], (-1,1,1,1,1)).shape, invx[odd_inds[:len(scales[i][1])]].shape, j0_.shape)
            print(scales[i][0].shape, invx[even_inds[:len(scales[i][0])]].shape, j0.shape)
            print(len(scales[i][0]), even_inds)
            res.append(np.tile(
                #j0*np.sum(np.exp(np.reshape(scales[i][0], (-1,1,1,1,1))+invx[even_inds[:len(scales[i][0])]]), 0)
                #+ j0_*np.sum(np.exp(np.reshape(scales[i][1], (-1,1,1,1,1))+invx[odd_inds[:len(scales[i][1])]]), 0),
                np.einsum('i,iabcd->abcd',
                  scales[i][0].astype(np.float64), invx[even_inds[:len(scales[i][0])]])*j0\
                + np.einsum('i,iabcd->abcd',
                  scales[i][1].astype(np.float64), invx[odd_inds[:len(scales[i][1])]])*j0_,
                (counts[i],1,1,1,1)))
            if x[0,0,0,0] == 0:
              if nn == 0:
                res[-1][:,:,0,:,:] = 1.
              else:
                res[-1][:,:,0,:,:] = 0


            print(res[-1].shape)
            #fig, ax = plt.subplots()
            #ax.plot(res[-1][0,0,:,0,0], '-k')
            #ax.plot(sp.special.spherical_jn(nn*2, x[0,:,0,0]), '--b')
            #fig.savefig("compare_jl{}.png".format(nn*2))

          return np.concatenate(res, 0)
        """

  def simulate_error_data(self):
    """
    This function introduces expiremental error, based on imported data,
    into the simulation by applying a high pass filter to the data and 
    adding this noise to the simulation, and using the imported variances
    from data.

        Parameters
        ----------

        Returns
        -------
    """

    # Check/calculate scaling between data and C simulation
    if "scale" not in self.data_params:
      fit_I0(self.input_data_coeffs,
          data=self.input_data_coeffs_, var=self.input_data_coeffs_var_)
    elif self.data_params["scale"] is None: 
      fit_I0(self.input_data_coeffs,
          data=self.input_data_coeffs_, var=self.input_data_coeffs_var_)
    else:
      self.I = np.array([[self.data_params["scale"]]])

    # Remove nans
    filt_inp = copy(self.input_data_coeffs_)
    i_, inds = np.arange(filt_inp.shape[0]), np.zeros(filt_inp.shape[0])
    while np.any(np.isnan(filt_inp[i_,ind])):
      ii = np.isnan(filt_inp[i_,ind])
      inds[ii] += 1
    for i in range(filt_inp.shape[0]):
      filt_inp[i,:inds[i]] = filt_inp[i,inds[i]+1]

    # Plot the fourier power spectrum and filter
    b, a = sp.signal.butter(*error_options, btype="hp", analog=True)
    w, h = sp.signal.freqs(b, a)
    self.plot_filter(filt_inp, (w, h), plot_folder)

    # Filter the data to get the noise
    sos = sp.signal.butter(*error_options, btype="hp", output='sos')
    self.experimental_noise = sp.signal.sosfiltfilt(sos, filt_inp)/self.I

    # Add noise to calculated signal
    self.input_data_coeffs += self.experimental_noise

    for il in range(self.input_data_coeffs.shape[0]):
      fig, axs = plt.subplots(2, 1, sharex=True,
          gridspec_kw={"height_ratios": [3, 1]})
      axs[0].plot(self.dom,
          (self.input_data_coeffs_/self.I-self.experimental_noise)[0,:],
          '--g', label="Noise Subtracted Data")
      axs[0].plot(self.dom, self.input_data_coeffs[0,:],
          '-k', label="Simulation with Data Noise")
      axs[0].plot(self.dom,
          (self.input_data_coeffs-self.experimental_noise)[0,:],
          '--b', label="Simulation Without Noise")
      axs[1].plot(self.dom, self.experimental_noise[0,:], '-k')
      axs[0].set_xlim([self.dom[0], self.dom[-1]])
      axs[1].set_xlim([self.dom[0], self.dom[-1]])
      axs[1].set_xlabel(r'q $[\AA^{-1}]$')
      axs[1].set_ylabel("C Noise [arb]")
      axs[0].set_ylabel("C [arb]")
      fig.legend()
      plt.tight_layout()
      fig.savefig(os.path.join(plot_folder,
          "sim_add_data_noise_lmk-{}-{}-{}.png".format(*self.data_LMK[il])))

    # Set the variance to the scaled data variance
    self.experimental_var = self.input_data_coeffs_var_/(self.I**2)

    # Reset I to 1 since we are using simulated data
    self.I = np.ones((1,1))


  def plot_filter(self, data, plot_filter, plot_folder, axs=None):
    """
    Plot the high pass and low pass filters applied in this analysis

        Parameters
        ----------
        data : 2D np.array of type float [lmk,q]
            The data which the high/low pass filter was applied to
        plot_filter : tuple of filter parameters (w, h)
            The w and h parameters of python filters
        plot_folder : string
            The folder address where the plots will be saved
        axs : list of pyplot subplot axes
            If None the axes will be made, if not the a list of axes for each
            LMK projection
        
        Returns
        -------
    """

    # Calculate fft and filter to plot 
    fft_inp = np.concatenate([data, np.flip(data, axis=-1)[:,:-1]], axis=-1)
    fft_out = fft(fft_inp)
    fft_freqs = fftfreq(fft_inp.shape[-1])
    imax = np.argmax(fft_freqs)
    
    for il in range(self.data_LMK.shape[0]):
      if axs is None:
        fig, ax = plt.subplots()
      else:
        ax = axs[il]
      ax.plot(fft_freqs[:imax+1], np.abs(fft_out[0,:imax+1])**2, '-k')
      ax2 = ax.twinx()
      ax2.plot(plot_filter[0], np.abs(plot_filter[1]), '-b')
      ax.set_xlim([0, fft_freqs[imax]])
      ax2.set_xlim([0, fft_freqs[imax]])
      ax2.set_ylim([0, 1.01])
      ax2.tick_params(axis='y', labelcolor='blue') 
      ax.set_yscale('log')
      ax.set_xlabel("Frequency [1/Nbins]")
      ax.set_ylabel("Power")
      ax2.set_ylabel("Filter")
      if axs is None:
        fig.savefig(os.path.join(plot_folder,
            "high_pass_filter_lmk-{}-{}-{}.png".format(*self.data_LMK[il])))


  def simulate_error_StoN(self, error_options):
    """
    This function simulates the C coefficient errors by adding Poissonian
    error to the diffraction pattern and propogating it through the fitting
    process.

        Parameters
        ----------
        error_options : tuple of type floats
            A tuple of the signal to noise ratio and the range it is calculated
            over. The signal to noise is calculated with C000 within the given
            range. (s2n_scale, s2n_range)

        Returns
        -------
    """

    s2n_scale, s2n_range = error_options

    # Get ADMs
    if self.ADMs is None:
      if "ADM_params" in self.data_params:
        _, self.ADMs, _, _ = self.get_ADMs(self.data_params["ADM_params"], 
            get_LMK=self.data_LMK)
      else:
        _, self.ADMs, _, _ = self.get_ADMs(self.data_LMK)

    # Calculate q map
    if self.data_params["dom"][0] == 0:
      N = self.data_params["dom"].shape[0]
      N_hole = 0
      q_wHole = self.data_params["dom"]
    else:
      dq = self.data_params["dom"][1] - self.data_params["dom"][0]
      N_hole = np.int(np.round(self.data_params["dom"][0]/dq))
      N_hole += (N_hole+1)%2
      dqq = self.data_params["dom"][0]/N_hole
      q_wHole = np.concatenate(
          [np.arange(N_hole)*dqq, self.data_params["dom"]], axis=0)

    qx, qy = np.meshgrid(
        np.concatenate([np.flip(q_wHole[1:]), q_wHole]),
        np.concatenate([np.flip(q_wHole[1:]), q_wHole]))
    q_map = np.sqrt(qx**2 + qy**2)

    # Account for K/-K double counting for bases
    azim_scale = []
    for lmk in self.data_params["fit_bases"]:
      if lmk[0] == 0:
        azim_scale.append(1)
      else:
        if [lmk[0], lmk[1], -1*lmk[2]] in self.data_params["fit_bases"]:
          azim_scale.append(1)
        else:
          azim_scale.append(2)
    azim_scale = np.array(azim_scale)

    if 0 not in self.data_LMK[:,0]:
      sim_LMK = np.concatenate([np.array([[0,0,0]]), self.data_LMK])
      azim_scale = np.concatenate([np.ones(1), azim_scale])
      sim_LMK_weights = np.concatenate(
          [np.ones((1, self.ADMs.shape[1]))/(8*np.pi**2),
          self.ADMs]).transpose()*azim_scale
    else:
      sim_LMK = self.data_LMK
      sim_LMK_weights = self.ADMs.transpose()*azim_scale

    # Simulate Diffraction
    if len(sim_LMK_weights.shape) == 2 and sim_LMK_weights.shape[0] > 10:
      mol_diffraction = []
      for i in range(np.int(np.ceil(sim_LMK_weights.shape[0]/10.))):
        atm_diffraction, mol = diffraction_calculation(
            sim_LMK, sim_LMK_weights[i*10:(i+1)*10],
            np.array([[self.atom_positions]]), [self.atom_types],
            self.scat_amps_interp, q_map, self.data_params["detector_dist"],
            freq=self.data_params["wavelength"])

        mol_diffraction.append(mol[0])
      mol_diffraction = np.concatenate(mol_diffraction, axis=0)
    else:
      atm_diffraction, mol_diffraction = diffraction_calculation(
          sim_LMK, sim_LMK_weights,
          np.array([[self.atom_positions]]), [self.atom_types],
          self.scat_amps_interp, q_map, self.data_params["detector_dist"],
          freq=self.data_params["wavelength"])
      mol_diffraction = mol_diffraction[0]

    total_diffraction = (atm_diffraction + mol_diffraction)\
        /atm_diffraction
    s2n_var = total_diffraction/atm_diffraction
    """
    for d in range(total_diffraction.shape[0]):
      pme = mol_diffraction[d]/atm_diffraction
      rng = np.amax([np.amax(pme), np.abs(np.amin(pme))])
      plt.pcolormesh(qx,qy, pme, vmax=rng, vmin=-1*rng, cmap='seismic')
      plt.colorbar()
      plt.savefig("testdiff{}.png".format(d))
      plt.close()
    """

    # Fit lab frame Ylm and propogate error
    N = q_wHole.shape[0]
    rx, ry = np.meshgrid(np.arange(N*2+5), np.arange(N*2+5))
    rad = np.round(np.sqrt((rx-rx.shape[1]//2)**2 + (ry-ry.shape[0]//2)**2))
    rad_inds = {}
    for rr in np.arange(N):
      if rr >= N_hole:
        rad_inds[rr] = np.where(rad == rr)
        rad_inds[rr] = (rad_inds[rr][0]-rx.shape[0]//2, rad_inds[rr][1]-rx.shape[0]//2)

    img_fits, img_covs = fit_legendres_images(
        mol_diffraction/atm_diffraction,
        np.array([[atm_diffraction.shape[0]//2, atm_diffraction.shape[1]//2]]\
            *s2n_var.shape[0]),
        np.concatenate([np.array([0]), np.unique(self.data_LMK[:,0])]),
        rad_inds, N, image_stds=np.sqrt(s2n_var),
        chiSq_fit=True)
        
    print("fin shapes", img_fits.shape, img_covs.shape)
    """
    for i in range(4):
      plt.pcolormesh(img_fits[:,4:,i])
      plt.colorbar()
      plt.savefig("testinfits{}.png".format(i))
      plt.close()
    """


    s2n_range = [
        np.argmin(np.abs(s2n_range[0]-self.data_params["dom"])),
        np.argmin(np.abs(s2n_range[1]-self.data_params["dom"]))]
    s2n_inds = np.arange(s2n_range[1]-s2n_range[0]) + s2n_range[0]
    s2n_inds_ = (img_covs[:,s2n_range[0]:s2n_range[1],0,0] != 0)
    s2n = np.nanmean(np.abs(
        img_fits[:,s2n_inds,0]/np.sqrt(img_covs[:,s2n_inds,0,0]))[s2n_inds_])
    inds = np.arange(img_covs.shape[-1])
    s2n_var = img_covs[:,:,inds,inds]*(s2n/s2n_scale)**2
    print("S2N VAR", s2n, s2n_scale, s2n_var.shape)
    print(s2n_var[:5])

    fit_var = np.zeros_like(self.input_data_coeffs)
    for il,l in enumerate(np.unique(self.data_LMK[:,0])):
      linds = (self.data_LMK[:,0] == l)
      print(self.data_LMK)
      print("IN L",l, linds)
      fit_ADMs = self.ADMs[linds,:]
      fit_ADMs -= np.mean(fit_ADMs, -1, keepdims=True)
      print("err shapes", self.ADMs.shape, fit_ADMs.shape, self.input_data_coeffs.shape)
      print("matrix",  np.einsum('ai,ib,ci->bac', 
        fit_ADMs, 1/s2n_var[:,:,il], fit_ADMs)[:5])
      fit_var_ = np.linalg.inv(
          np.einsum('ai,ib,ci->bac', 
            fit_ADMs, 1/s2n_var[:,:,il], fit_ADMs))
      ev_inds = np.arange(fit_var_.shape[-1])
      fit_var[linds,:] = np.transpose(fit_var_[:,ev_inds,ev_inds])
    s2n_var = fit_var

    if not self.data_params["isMS"]:
      s2n_var *= self.atm_scat**2

    if self.experimental_var is None:
      self.experimental_var = s2n_var
    else:
      self.experimental_var += s2n_var




  ########################
  #####  Playground  #####
  ########################

  def calc_principleI(self, I_tensor):
    """
    I_tensor: Moment of inertia tensor
    Calculate the principal moments of inertia by solving for 
    the eigenvalues of I_tensor.
    Code taken from: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Normal,_Hermitian,_and_real-symmetric_matrices
    """
    
    p1 = I_tensor[0,1]**2 + I_tensor[0,2]**2 + I_tensor[1,2]**2
    q = np.trace(I_tensor)/3.
    p2 = (I_tensor[0,0] - q)**2 + (I_tensor[1,1] - q)**2\
        + (I_tensor[2,2] - q)**2 + 2*p1
    p = np.sqrt(p2/6.)
    B = (1./p)*(I_tensor-q*np.eye(3)) 
    r = np.linalg.det(B)/2.

    phi = np.arccos(np.min([np.max([-1, r]), 1]))/3.

    I0 = q + 2*p*np.cos(phi)
    I2 = q + 2*p*np.cos(phi + (2*np.pi/3.))
    I1 = 3*q - I0 - I2

    print("COMPARE", np.linalg.eig(I_tensor))
    return np.array([I0, I1, I2])


  def calc_principal_axis(self, I, Ip):
    """
    I: Moment of intertia tensor
    Ip: array of principal moment of inertia to find association vector
    Calculation the principal axis unit vector.
    """
    
    a = -1*(I[0,2]*I[2,1] - I[0,1]*(I[2,2] - Ip))\
        /(I[2,1]*(I[0,0] - Ip) - I[0,1]*I[2,0])
    eig_vec = np.array([a, (I[1,2] - I[1,0]*a)/(I[1,1] - Ip), np.ones_like(Ip)])#.transpose()
    print("I tens", I)
    print("VALS", Ip)
    print("EIGS", eig_vec)
    print("gamma", np.sqrt(1./(1+(I[1,2]-I[1,0]*a)**2 + a**2)))
    return eig_vec/np.linalg.norm(eig_vec, axis=0, keepdims=True)


  #######################
  #####  Graveyard  #####
  #######################
  """
  def get_molecule_perturber(self):

    shape = self.atom_positions.shape

    if self.data_params["molecule"] == "N2O":
      def perturb_molecule(prev_molecule):
        perturbation = np.random.uniform(-1,1, 3)
        perturbation *= self.data_params["perturb_range"]

        rNO = np.linalg.norm(prev_molecule[2,:])
        angle = np.arccos(np.sum(prev_molecule[0,:]*prev_molecule[2,:])\
            /(rNO*prev_molecule[0,2]))
        rNO += perturbation[1] 
        angle += perturbation[2] 
        
        molecule = np.zeros_like(prev_molecule)
        molecule[0,2] = prev_molecule[0,2] + perturbation[0]
        molecule[2,0] = rNO*np.sin(angle)
        molecule[2,2] = rNO*np.cos(angle)
        
        return self.rotate_to_principalI(molecule)

    elif self.data_params["molecule"] == "NO2":
      def perturb_molecule(prev_molecule):
        perturbation = np.random.uniform(-1,1, 3)
        perturbation *= self.data_params["perturb_range"]

        rNO1 = np.linalg.norm(prev_molecule[0,:] - prev_molecule[1,:])
        rNO2 = np.linalg.norm(prev_molecule[2,:] - prev_molecule[1,:])
        angle = np.arccos(
            np.sum((prev_molecule[0,:] - prev_molecule[1,:])\
                *(prev_molecule[2,:]  - prev_molecule[1,:]))/(rNO1*rNO2))
        rNO1 += perturbation[0] 
        rNO2 += perturbation[1] 
        angle += perturbation[2] 
        
        molecule = np.zeros_like(prev_molecule)
        molecule[0,0] = rNO1*np.cos(angle/2)
        molecule[0,2] = rNO1*np.sin(angle/2)
        molecule[2,0] = rNO2*np.cos(angle/2)
        molecule[2,2] = -1*rNO2*np.sin(angle/2)

        return self.rotate_to_principalI(molecule)

    else:
      def perturb_molecule(molecule):
        perturbation = np.random.uniform(
            -1*self.data_params["perturb_range"],
            self.data_params["perturb_range"],
            shape)

        return molecule + perturbation

    return perturb_molecule
  
  
  def calculate_coeffs_lmk(self, R, lmk):

    raise NotImplementedError("fix use of calc coeffs")
    temp_data_Lcalc = self.data_Lcalc
    temp_data_Mcalc = self.data_Mcalc
    temp_data_Kcalc = self.data_Kcalc

    self.data_Lcalc = np.reshape(lmk[:,0], (-1, 1, 1))
    self.data_Mcalc = np.reshape(lmk[:,1], (-1, 1, 1))
    self.data_Kcalc = np.reshape(lmk[:,2], (-1, 1, 1))

    calc_coeffs = self.calculate_coeffs(R)
    
    self.data_Lcalc = temp_data_Lcalc
    self.data_Mcalc = temp_data_Mcalc
    self.data_Kcalc = temp_data_Kcalc

    return calc_coeffs

  """


