import sys, os, glob, time
import errno
import h5py
import emcee
import corner
from copy import copy as copy
import numpy as np
import scipy as sp
import numpy.random as rnd
from multiprocessing import Pool
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from modules.spherical_j_cpp import spherical_j

sys.path.append("/cds/home/k/khegazy/simulation/diffractionSimulation/modules")
from diffraction_simulation import diffraction_calculation
sys.path.append("/cds/home/k/khegazy/baseTools/modules")
from fitting import fit_legendres_images

#import jax as jax
#import jax.numpy as jnp


"""
def dists_(r1, r2):
  R = r1 - r2
  r = jnp.sqrt(jnp.sum(R**2))
  theta = jnp.arccos(R[2]/(r + 1e-20))
  phi   = jnp.arctan2(R[1],R[0])
  return jnp.array([r, theta, phi])

def calc_dists_vm(R):
  return jax.vmap(jax.vmap(dists_, in_axes=(None, 0)),
    in_axes=(0, None))(R,R)
"""



def calc_dists(R):
  r     = np.expand_dims(R, 1) - np.expand_dims(R, 0)
  dR    = np.sqrt(np.sum(r**2, axis=2))
  theta = np.arccos(r[:,:,2]/(dR + 1e-20))
  phi   = np.arctan2(r[:,:,1], r[:,:,0])

  return np.concatenate([np.expand_dims(dR,2),\
    np.expand_dims(theta, 2),\
    np.expand_dims(phi, 2)], axis=2)

def calc_all_dists(R):
  r     = np.expand_dims(R, 2) - np.expand_dims(R, 1)
  dR    = np.sqrt(np.sum(r**2, axis=-1))
  theta = np.arccos(r[:,:,:,2]/(dR + 1e-20))
  phi   = np.arctan2(r[:,:,:,1], r[:,:,:,0])

  return np.concatenate([np.expand_dims(dR,-1),\
    np.expand_dims(theta, -1),\
    np.expand_dims(phi, -1)], axis=-1)


class density_extraction:

  def __init__(self, data_params,
      get_scattering_amplitudes,
      log_prior=None,
      theta_to_cartesian=None,
      ensemble_generator=None,
      get_ADMs=None,
      results_only=False):

    self.sampler = None
    self.has_converged = False
    self.tau_convergence = None

    self.data_params = data_params
    self.ensemble_generator = ensemble_generator
    self.atom_info = {
      "H" : ["hydrogen", 1.0],
      "C" : ["carbon", 12.0],
      "O" : ["oxygen", 16.0],
      "N" : ["nitrogen", 14.0],
      "F" : ["flourine", 19.0],
      "Br": ["bromine", 78.0],
      "I" : ["iodine", 127.0]
    }

    if "output_dir" not in self.data_params:
      self.data_params["output_dir"] = "output"

    self.I = np.ones((1,1))
    self.do_rebin = False

    # Log Prior
    if log_prior is not None:
      self.log_prior = log_prior
    else:
      self.log_prior = self.default_log_prior

    # Convert between MCMC parameters theta to cartesian
    if theta_to_cartesian is not None:
      self.theta_to_cartesian = theta_to_cartesian

    # Gather ADMs for simulated error
    if get_ADMs is not None:
      self.get_ADMs = get_ADMs

    # Plotting
    self.plot_setup = True
    if "plot_setup" in data_params:
      self.plot_setup = data_params["plot_setup"]

    # Skip if results only
    if results_only:
      fileName = os.path.join(self.data_params["output_dir"], self.get_fileName() + ".h5")
      self.load_emcee_backend(fileName)

      return

    # Make output folders
    print("Making output folder")
    if not os.path.exists(os.path.join(
        self.data_params["output_dir"], self.get_fileName(folder_only=True))):
      try:
        os.makedirs(os.path.join(
            self.data_params["output_dir"], self.get_fileName(folder_only=True)))
      except OSError as e:
        if e.errno != errno.EEXIST:
          raise
    if not os.path.exists(os.path.join("plots", self.get_fileName(folder_only=True))):
      try:
        os.makedirs(os.path.join("plots", self.get_fileName(folder_only=True)))
      except OSError as e:
        if e.errno != errno.EEXIST:
          raise

    print("posterior type")
    # Setup posterior type
    self.log_likelihood = self.log_likelihood_optimal
    self.C_log_likelihood = self.gaussian_log_likelihood
    if "posterior" in self.data_params:
      if "ensity" in self.data_params["posterior"]:
        self.log_likelihood = self.log_likelihood_density
    else:
      self.data_params["posterior"] = "optimal"

    print("getting init geo")
    # Get initial geometry
    self.get_molecule_init_geo()
    
    print("Setting up mom inert")
    # Setup moment of inertia tensor calculation
    self.setup_I_tensor_calc()

    print("getting atom pos")
    # Rotate initial geometry into the MF
    self.atom_positions = self.rotate_to_principalI(self.atom_positions)

    # Get data
    print("getting data")
    self.data_or_sim = True
    if "simulate_data" in self.data_params:
      if self.data_params["simulate_data"]:
        self.data_or_sim = False
    self.get_data()

    # Get scattering amplitudes
    print("Getting Scat Amps")
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

    # Get Spherical Bessel function calculation
    self.spherical_j = self.get_spherical_j_calc(self.data_LMK[:,0])

    # Simulate data if needed
    print("simulate data")
    self.data_Lcalc = np.reshape(self.data_LMK[:,0], (-1, 1, 1))
    self.data_Mcalc = np.reshape(self.data_LMK[:,1], (-1, 1, 1))
    self.data_Kcalc = np.reshape(self.data_LMK[:,2], (-1, 1, 1))
    if not self.data_or_sim:
      self.simulate_data()

    print("pruning data")
    # Prune data in time and dom
    self.prune_data()

    print("making weiner weight")
    # Calculate Wiener mask
    self.make_wiener_weight()

    
    self.eval_data_coeffs = self.data_coeffs
    self.eval_data_coeffs_var = self.data_coeffs_var
    self.eval_dom = self.dom

    """
    # Rebin Data
    self.do_rebin = False
    if "binning" in self.data_params:
      if self.data_params["binning"] is not None:
        self.do_rebin = True
        if self.data_params["binning"] == "log":
          Nrebin = np.cumsum(np.log2(np.arange(len(self.dom))+1).astype(int)) + 1
          Nrebin[1:] += 1
          prev = 0
          rebin_mat = []
          for Nb in Nrebin:
            rebin_mat.append(np.zeros_like(self.dom))
            rebin_mat[-1][prev:Nb] = 1./(np.min([Nb, len(self.dom)])-prev)
            prev = Nb
            if Nb >= len(self.dom):
              break
          self.rebin_mat = np.transpose(np.array(rebin_mat))
      self.eval_data_coeffs = np.matmul(self.data_coeffs, self.rebin_mat)
      self.eval_data_coeffs_var = np.matmul(self.data_coeffs_var, self.rebin_mat)
      self.eval_dom = np.matmul(self.dom, self.rebin_mat)
    else:
      self.eval_data_coeffs = self.data_coeffs
      self.eval_data_coeffs_var = self.data_coeffs_var
      self.eval_dom = self.dom
    """

    # Subtract the mean from the data
    # TODO Fix mean subtraction or whatever it should be | Same for C_dist
    #self.eval_data_coeffs -= np.expand_dims(np.mean(
    #    self.eval_data_coeffs[:,:], axis=-1), -1)


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
    print("DONE J", self.data_LMK[:,0])
    #print(j_check[0,0,75:100,0,0])
    #print(sp.special.spherical_jn(2, self.dom)[75:100])
    for n in np.unique(self.data_LMK[:,0]):
      ii = np.where(self.data_LMK[:,0] == n)[0][0]
      std_check = np.abs(sp.special.spherical_jn(n, self.dom)-j_check[ii,0,:,0,0])\
          /np.sqrt(self.eval_data_coeffs_var[ii])
      m = np.amax(std_check)
      if m < 1.:
        mm = "Passed"
      else:
        mm = "FAILED AT Q={}".format(self.dom[np.argmax(std_check)])
      print("L = {} \t {} std ... {}  {}".format(n, m, mm, ii))

      # Plot Comparison
      fig, ax = plt.subplots()
      ax.plot(self.dom, j_check[ii,0,:,0,0], '-k')
      ax.plot(self.dom, sp.special.spherical_jn(n, self.dom), '--b')
      ax.set_xlabel("jn(q)")
      ax.set_xlim(self.dom[0], self.dom[-1])
      ax1 = ax.twinx()
      ax1.plot(self.dom, std_check)
      ax1.set_xlabel("residual [std]")
      #ax1.plot(px, (res[i][0,cut:,0,0]-sp.special.spherical_jn(i*2, x[0,cut:,0,0]))/sp.special.spherical_jn(i*2, x[0,cut:,0,0]))
      fig.savefig("check_jl{}_calculation.png".format(n))




  def theta_to_cartesian(self, theta):
    """
    Converts MCMC parameters to cartesian coordinates

    Args:
        theta: MCMC parameters [Nwalkers, Nparams]

    Returns:
        Cartesian representation of each molecule [Nwalkers, Natoms, 3]
    """

    return theta


  def get_ADMs(self, data_LMK, kwargs=None):
    """
    Get ADMs if they are not saved with the input data as is default. ADMs
    are only needed when using simulated error bars.

    Args:
        kwargs: key word args for your custom function

    Returns:
        numpy array of ADMs in the same order of self.data_LMK
    """

    raise NotImplementedError("Must implement get_ADMs")


  def setup_I_tensor_calc(self):

    self.mass = np.array([self.atom_info[t][1] for t in self.atom_types])
    self.mass = np.reshape(self.mass, (-1,1,1))
    self.Itn_inds1, self.Itn_inds2 = np.array([2,0,1]), np.array([1,2,0])
    self.Itn_idiag = np.arange(3)

  
  def calc_I_tensor(self, R):
    # Off diagonal terms
    I_tensor = -1*np.expand_dims(R, -1)*np.expand_dims(R, -2)

    # Diagonal terms
    I_tensor[:,self.Itn_idiag,self.Itn_idiag] =\
        R[:,self.Itn_inds1]**2 + R[:,self.Itn_inds2]**2

    return np.sum(I_tensor*self.mass, 0)

  
  def calc_I_tensor_ensemble(self, R):
    # Off diagonal terms
    I_tensor = -1*np.expand_dims(R, -1)*np.expand_dims(R, -2)

    # Diagonal terms
    I_tensor[:,:,self.Itn_idiag,self.Itn_idiag] =\
        R[:,:,self.Itn_inds1]**2 + R[:,:,self.Itn_inds2]**2

    return np.sum(I_tensor*self.mass, -3)




  def get_data(self):
    # Get the variance, measurement degree, legendre inds from fit results
    self.ADMs = None
    if "data_fileName" in self.data_params:
      with h5py.File(self.data_params["data_fileName"], "r") as h5:
        self.diffraction_LMK = h5["data_LMK"][:]
        self.data_LMK, self.input_data_coeffs_var = [], []
        for i in range(self.diffraction_LMK.shape[0]):
          self.data_LMK.append(
              h5["fit_LMK_dataLMKindex-{}".format(i)][:].astype(int))
          cov_inds = np.arange(self.data_LMK[-1].shape[0])
          self.input_data_coeffs_var.append(np.transpose(
              h5["fit_coeffs_cov_dataLMKindex-{}".format(i)][:][:,cov_inds,cov_inds]))
        #self.input_data_coeffs_var *= np.expand_dims(self.input_data_coeffs_var[:,70,:,:    ], -1)
        #self.data_lg = h5["legendre_inds"][:]
        self.dom = h5["fit_axis"][:]*self.data_params["q_scale"]
        if self.data_or_sim:
          self.input_data_coeffs = []
          for i in range(self.diffraction_LMK.shape[0]):
            self.input_data_coeffs.append(np.transpose(
                h5["fit_coeffs_dataLMKindex-{}".format(i)][:]))
          self.input_data_coeffs = np.concatenate(self.input_data_coeffs, axis=0)
      self.data_LMK = np.concatenate(self.data_LMK, axis=0)
      self.input_data_coeffs_var = np.concatenate(self.input_data_coeffs_var, axis=0)
    else:
      self.data_LMK = self.data_params["fit_bases"]
      self.input_data_coeffs_var = None
      self.dom = self.data_params["dom"]
      print("INP DOM", self.dom.shape)


    self.dom_mask = np.ones_like(self.dom).astype(bool)
    if "fit_range" in self.data_params:
      self.data_params["fit_range"][1] =\
          np.min([self.data_params["fit_range"][1], self.dom[-1]])
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

      self.dom_mask[self.dom<self.data_params["fit_range"][0]] = False
      self.dom_mask[self.dom>self.data_params["fit_range"][1]] = False

    if self.data_or_sim:
      ind2 = np.where(
          (self.data_LMK[:,0] == 2) & (self.data_LMK[:,2] == 0))[0]
      SN_ratio_lg2 = np.nanmean(
          self.input_data_coeffs[ind2,self.dom_mask]**2\
            /self.input_data_coeffs_var[ind2,self.dom_mask])
      #print("SN data ratio L=2: {} {}".format(
      #    SN_ratio_lg2, self.data_params["fit_range"]))


  def simulate_data(self):
    molecule = self.setup_calculations(skip_scale=True, plot=False)
    self.C_distributions = []
    
    if self.ensemble_generator is not None:
      ensemble = self.ensemble_generator(
          np.expand_dims(np.array(self.data_params["sim_thetas"]), 0))
      if len(ensemble) == 2:
        ensemble, weights = ensemble
      else:
        weights = np.ones((ensemble.shape[0], 1))

      self.input_data_coeffs = np.zeros(
          (self.data_LMK[:,0].shape[0], self.dom.shape[0]))
      self.input_data_coeffs_var = None

      ind, ind_step = 0, int(np.ceil(100000./ensemble.shape[0]))
      while ind < ensemble.shape[1]:
        ind_end = np.min([ensemble.shape[1], ind + ind_step])
        #calc_coeffs = self.calculate_coeffs_ensemble(ensemble)
        calc_coeffs = self.calculate_coeffs_ensemble(ensemble[:,ind:ind_end])
        self.input_data_coeffs = self.input_data_coeffs\
            + np.sum(np.sum(calc_coeffs\
            *np.expand_dims(np.expand_dims(weights[:,ind:ind_end], -1), -1),
            0), 0)
        self.C_distributions.append(calc_coeffs)
        ind = ind_end
 
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
      test = self.calculate_coeffs(molecule)

      """
      for i in range(test.shape[0]):
        plt.plot(test[i], '-k')
        plt.plot(calc_coeffs[0,i], '-b')
        plt.plot(self.input_data_coeffs[i], 'r')
        plt.savefig("compare_{}.png".format(i))
        plt.close()
      """
      #plt.errorbar(self.eval_dom, self.eval_data_coeffs[i,:], np.sqrt(self.eval_data_coeffs_var[i,:]))
    else:
      self.input_data_coeffs = self.calculate_coeffs(molecule)
   
      if not self.data_params["isMS"]:
        self.input_data_coeffs *= self.atm_scat
      
    # Simulate Error
    self.experimental_var = None
    if "simulate_error" in self.data_params:
      
      error_type, error_options = self.data_params["simulate_error"]
      
      ###  Generate errors for the total signal  ###
      if "StoN" in error_type:
        s2n_scale, s2n_range = error_options

        # Get ADMs
        if self.ADMs is None:
          if "ADM_kwargs" in self.data_params:
            self.ADMs = self.get_ADMs(
                self.data_LMK, kwargs=self.data_params["ADM_kwargs"], normalize=False)
          else:
            self.ADMs = self.get_ADMs(self.data_LMK)

        # Calculate q map
        if self.data_params["dom"][0] == 0:
          N = self.data_params["dom"].shape[0]
          N_hole = 0
          q_wHole = self.data_params["dom"]
        else:
          dq = self.data_params["dom"][1] - self.data_params["dom"][0]
          N_hole = np.int(np.round(self.data_params["dom"][0]/dq))
          N_hole += (N_hole+1)%2
          print("ADDING HOLE", dq, self.data_params["dom"][0], N_hole)
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
                np.array([[molecule]]), [self.atom_types], self.scat_amps_interp,
                q_map, self.data_params["detector_dist"],
                freq=self.data_params["wavelength"])

            mol_diffraction.append(mol[0])
          mol_diffraction = np.concatenate(mol_diffraction, axis=0)
        else:
          atm_diffraction, mol_diffraction = diffraction_calculation(
              sim_LMK, sim_LMK_weights,
              np.array([[molecule]]), [self.atom_types], self.scat_amps_interp,
              q_map, self.data_params["detector_dist"],
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

        print("zero", np.where(atm_diffraction == 0), np.where(np.sqrt(s2n_var) == 0))
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
        s2n = np.mean(np.abs(
            img_fits[:,s2n_range[0]:s2n_range[1],0]\
            /np.sqrt(img_covs[:,s2n_range[0]:s2n_range[1],0,0])))
        inds = np.arange(img_covs.shape[-1])
        s2n_var = img_covs[:,:,inds,inds]*(s2n/s2n_scale)**2
        print("ASFASDF", s2n_var[:5])

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

      elif "onstant_background" in error_type:
        variance_scale = error_options
        self.experimental_var = np.ones(self.input_data_coeffs.shape[-1], dtype=np.float)
        if self.data_params["isMS"]:
          self.experimental_var /= self.atm_scat**2
        
        # Error scale derived from input signal
        ind2 = (self.data_LMK[:,0] == 2)*(self.data_LMK[:,2] == 0)
        if self.input_data_coeffs_var is None:
          SN_ratio_lg2 = np.nanmean(
              self.input_data_coeffs[ind2,self.dom_mask]**2\
                /self.experimental_var[self.dom_mask])
        else:
          SN_ratio_lg2 = np.nanmean(
              self.input_data_coeffs[ind2,self.dom_mask]**2\
                /self.input_data_coeffs_var[ind2,self.dom_mask])
        #print("SCALE", self.data_params["simulate_error_scale"], SN_ratio_lg2/self.data_params["simulate_error_scale"])

        # Combine error scale with input scale factor
        if variance_scale is None:
          self.data_params["simulate_error"] = (error_type, 1.)
          variance_scale = 1.
        variance_scale *= SN_ratio_lg2

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





  def prune_data(self):
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
    self.data_coeffs = self.data_coeffs[:,self.dom_mask]
    self.data_coeffs_var = self.data_coeffs_var[:,self.dom_mask]
    #if self.C_distributions is not None:
    #  self.C_distributions = temp_C_distributions_LMK[:,self.dom_mask,:]
    self.atm_scat = self.atm_scat[self.dom_mask]
    if self.experimental_var is not None:
      self.experimental_var = self.experimental_var[:,self.dom_mask]
    for atm in self.scat_amps.keys():
      self.scat_amps[atm] = self.scat_amps[atm][self.dom_mask] 
    self.dist_sms_scat_amps = self.dist_sms_scat_amps[:,:,self.dom_mask]
   

  def make_wiener_weight(self):
    # Wiener Filter
    if self.experimental_var is not None:
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


      N = copy(self.experimental_var)
      if not self.data_params["isMS"]:
        N /= self.atm_scat**2

      self.wiener = S/(S+N)
    else:
      self.wiener = np.ones_like(self.data_coeffs)

    if self.plot_setup:
      for lg in range(len(self.data_LMK)):
        fig, ax = plt.subplots()
        handles = []
        handles.append(ax.errorbar(self.dom, self.data_coeffs[lg,:],\
            np.sqrt(self.data_coeffs_var[lg,:]),\
            label="legendre {}-{}-{}".format(*self.data_LMK[lg])))
        ax.plot(self.dom, self.data_coeffs[lg,:], '-k')
        ax2 = ax.twinx()
        ax2.plot(self.dom, self.wiener[lg,:], color='gray')
        ax2.tick_params(axis='y', labelcolor='gray') 
        ax.legend(handles=handles)
        fig.savefig("./plots/{}/data_coeffs_lg-{}-{}-{}.png".format(
          self.data_params["molecule"], *self.data_LMK[lg]))
        plt.close()
        #if self.data_LMK[lg][0] == 2 and self.data_LMK[lg][-1] == 0:
        #  print(self.data_coeffs[lg,:])

    # Must normalize filter to account for the removed bins
    w_norm_ = np.mean(self.wiener, axis=-1, keepdims=True)
    w_norm = 1./copy(w_norm_)
    w_norm[w_norm_==0] = 0
    self.wiener *= w_norm


  def evaluate_scattering_amplitudes(self):
    self.atm_scat = np.zeros_like(self.dom)
    self.scat_amps = {}
    for atm in self.atom_types:
      if atm not in self.scat_amps:
        self.scat_amps[atm] = self.scat_amps_interp[atm](self.dom)
      self.atm_scat += self.scat_amps[atm]**2


  def rotate_to_principalI(self, R):

    # Center of Mass
    R -= np.sum(R*self.mass[:,0], -2)/np.sum(self.mass)

    # Calculate principal moment of inertia vectors
    I_tensor = self.calc_I_tensor(R)
    Ip, I_axis = np.linalg.eigh(I_tensor)

    return np.matmul(R, I_axis)[:,np.array([1,2,0])]


  def rotate_to_principalI_ensemble(self, R):

    # Center of Mass
    R -= np.expand_dims(np.sum(R*self.mass[:,0], -2)/np.sum(self.mass), -1)

    # Calculate principal moment of inertia vectors
    I_tensor = self.calc_I_tensor_ensemble(R)
    Ip, I_axis = np.linalg.eigh(I_tensor)

    return np.matmul(R, I_axis)[:,:,np.array([1,2,0])]


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
          print(vals)
          self.atom_types.append(vals[0])
          pos = [float(x) for x in vals[1:]]
          self.atom_positions.append(np.array([pos]))

    self.atom_positions = np.concatenate(self.atom_positions, axis=0)


  
  def calculate_coeffs_lmk(self, R, lmk):
    """
    Calculate only the C coefficients for the specified lmk values
    """

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

  def calculate_coeffs(self, molecules):

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

    # Rebin
    if self.do_rebin:
      calc_coeffs = np.matmul(calc_coeffs, self.rebin_mat)

    # Subtract mean and normalize 
    # TODO Fix subtract mean
    #calc_coeffs -= np.expand_dims(np.mean(calc_coeffs, axis=-1), -1)
    calc_coeffs *= self.I

    return calc_coeffs

 
  def calculate_coeffs_ensemble(self, R):

    # Rotate molecule into the MF (Principal axis of I)
    #R = self.rotate_to_principalI_ensemble(R).transpose((1,2,0))
    tic = time.time()
    R = self.rotate_to_principalI_ensemble(R).transpose((2,3,0,1))
    print("\trotate time:", time.time()-tic)

    # Calculate pair-wise vectors
    all_dists = calc_dists(R)
    dists = all_dists[self.dist_inds]

    # Calculate diffraction response
    ttic = time.time()
    tic = time.time()
    C = np.complex(0,1)**self.data_Lcalc*8*np.pi**2/(2*self.data_Lcalc + 1)\
        *np.sqrt(4*np.pi*(2*self.data_Lcalc + 1))
    print("\tC time:", C.shape, time.time()-tic)
    tic = time.time()
    #J = sp.special.spherical_jn(
    #    np.expand_dims(np.expand_dims(self.data_Lcalc, -1), -1),
    #    np.expand_dims(np.expand_dims(self.calc_dom, -1), -1)\
    #    *np.expand_dims(dists[:,0], axis=1))
    inp = np.expand_dims(np.expand_dims(self.calc_dom, -1), -1)\
        *np.expand_dims(dists[:,0], axis=1)
    J = self.spherical_j(inp)
    #    np.expand_dims(np.expand_dims(self.calc_dom, -1), -1)\
    #    *np.expand_dims(dists[:,0], axis=1))
    print("\tJ time:", inp.shape, J.shape, time.time()-tic)
    tic = time.time()
    Y = sp.special.sph_harm(
        -1*np.expand_dims(np.expand_dims(self.data_Kcalc, -1), -1),
        np.expand_dims(np.expand_dims(self.data_Lcalc, -1), -1),
        np.expand_dims(np.expand_dims(dists[:,2], axis=0), axis=2),
        np.expand_dims(np.expand_dims(dists[:,1], axis=0), axis=2))
    print("\tY:", Y.shape, time.time()-tic)
    print("\tCJY time:", time.time()-ttic)

    # Sum all pair-wise contributions
    tic = time.time()
    calc_coeffs = np.sum(np.real(
        np.expand_dims(np.expand_dims(self.dist_sms_scat_amps, -1), -1)\
        *np.expand_dims(np.expand_dims(C, -1), -1)*J*Y), axis=1)
    print("\tsum time:", time.time()-tic)

    #plt.hist(calc_coeffs[-1,2,:], bins=25, weights=w[:,0])
    #plt.savefig("testDist.png")
    #plt.close()
    # Subtract mean and normalize 
    #calc_coeffs -= np.expand_dims(np.mean(calc_coeffs[:,:], axis=1), 1)
    calc_coeffs *= self.I

    return calc_coeffs.transpose((2,3,0,1))
 


  """
  def calculate_log_prob_density(self, R, n=0):

    calc_coeffs = self.calculate_coeffs(R)
   
    prob = np.mean(-0.5*(self.eval_data_coeffs - calc_coeffs)**2\
        /self.eval_data_coeffs_var)
    #    + np.log(1/np.sqrt(self.data_coeffs_var)))
   
    return prob
  """
  
  def default_log_prior(self, theta):
    return np.zeros(theta.shape[0])

  def gaussian_log_likelihood(self, calc_coeffs):
    return -0.5*(self.eval_data_coeffs - calc_coeffs)**2\
        /self.eval_data_coeffs_var

  def log_likelihood_density(self, theta, n=0):

    # Convert parameters to cartesian coordinates
    molecules = self.theta_to_cartesian(theta)

    calc_coeffs = self.calculate_coeffs_ensemble(molecules)

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

    # Simulate the molecular ensemble
    tic = time.time()
    print("INP theta shape",theta.shape)
    ensemble = self.ensemble_generator(theta)   # Cartesian
    print("OUT theta shape",theta.shape)
    if len(ensemble) == 2:
      ensemble, weights = ensemble
    else:
      weights = np.ones((ensemble.shape[0], 1))
    print("Ensemble time:", time.time()-tic)

    calc_coeffs = np.zeros(
        (theta.shape[0], self.data_LMK[:,0].shape[0], self.dom.shape[0]))

    ind, ind_step = 0, int(np.ceil(100000./ensemble.shape[0]))
    tic = time.time()
    while ind < ensemble.shape[1]:
      ind_end = np.min([ensemble.shape[1], ind + ind_step])
      calc_coeffs_ = self.calculate_coeffs_ensemble(ensemble[:,ind:ind_end])
      calc_coeffs = calc_coeffs\
          + np.sum(calc_coeffs_\
            *np.expand_dims(np.expand_dims(weights[:,ind:ind_end], -1), -1), 1)
      print("temp", ind, ind_step, ensemble[:,ind:ind_end].shape, calc_coeffs_.shape, calc_coeffs.shape)
      ind = ind_end

    print("calc time:", time.time()-tic)
    
    prob = np.nansum(np.nansum(
          self.wiener*self.C_log_likelihood(calc_coeffs),
        axis=-1), axis=-1)
    #    + np.log(1/np.sqrt(self.data_coeffs_var)))
 
    return prob


  def log_probability(self, theta):

    # Evaluate log prior
    lprior = self.log_prior(theta)

    # Evaluate log likelihood
    llike = self.log_likelihood(theta)

    return lprior + llike



  def setup_sampler(self, nwalkers=None, ndim=None, expect_file=False):

    fileName = os.path.join(self.data_params["output_dir"], self.get_fileName() + ".h5")
    print("FNAME", fileName)

    exists = os.path.exists(fileName)
    if (not exists and expect_file):
      raise RuntimeError(
          "ERROR: Expected file {} cannot be found!!!".format(fileName))

    if not exists and (nwalkers is None or ndim is None):
      raise RuntimeError(
          "ERROR: Expected file {} cannot be found and nwalkers or ndim is None!!!".format(fileName))

    last_walker_pos = None
    if exists:
      backend, last_walker_pos = self.load_emcee_backend(fileName)
      nwalkers, ndim = last_walker_pos.shape
    else:
      print("INFO: {} does not exist, creating new backend".format(fileName))
      backend = emcee.backends.Backend()
   
    print("Setting up MCMC")
    self.sampler = emcee.EnsembleSampler(
        nwalkers, ndim, self.log_probability,
        backend=backend, vectorize=True)

    return exists, last_walker_pos


  def run_mcmc(self, walker_init_pos, Nsteps):

    nwalkers, ndim = walker_init_pos.shape
    _, last_walker_pos = self.setup_sampler(nwalkers, ndim)

    print("Running MCMC")
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

      if not np.isnan(autocorr):
        # Plot Progress
        self.plot_emcee_results(
          self.sampler.get_chain(discard=2*int(autocorr),
              thin=int(autocorr)))

        # Check convergence
        conv1 = np.all(tau*100 < self.sampler.iteration)
        conv1 = conv1 + (np.all(tau*20 < self.sampler.iteration) and np.max(tau) > 250)
        conv2 = np.all(np.abs(self.tau_convergence[-2] - tau)/tau < 0.01)
        if "min_acTime_steps" in self.data_params:
          sample_limit = np.all(tau*self.data_params["min_acTime_steps"]\
              < self.sampler.iteration)


        self.has_converged = conv1*conv2 or self.has_converged
        
        print("Sample {}: mean tau = {} / convergence {} {}".format(
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

    if not self.has_converged:
      print(self.get_fileName())
    return backend, backend.chain[-1]


  def save_emcee_backend(self):
   
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

      max_ac = int(np.amax(tau))
      if self.sampler.backend.iteration > 4*max_ac:
        h5.create_dataset("filtered_chain",
            data=self.sampler.get_chain(discard=3*max_ac, thin=max_ac))
      else:
        h5.create_dataset("filtered_chain", data=np.array([False]))

  
  def get_mcmc_results(self, labels=None, plot=True):
    """
    Plot the trajectories of each walker and the correlation between
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
    max_ac = np.amax(tau).astype(int)//2 + 1

    filtered_chain = self.sampler.get_chain(
        discard=3*max_ac, thin=max_ac)
    if plot:
      self.plot_emcee_results(filtered_chain, labels=labels)

    return np.reshape(filtered_chain, (-1, filtered_chain.shape[-1]))


  def plot_emcee_results(self, samples, labels=None):
    if samples is None:
      print("WARNING: Cannot plot with samples = None, Skipping!!!")
      return

    fileName_base = os.path.join("./plots", self.get_fileName())
    #####  Plot Trajectory in Theta of Each Walker  #####
    #samples = self.sampler.get_chain()
    print("SAMPLE", samples.shape, self.get_fileName())
    Nsamples = samples.shape[-1]
    if labels is None and "labels" not in self.data_params:
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


  def setup_calculations(self, skip_scale=False, plot=True):
    self.calc_dom = np.expand_dims(self.dom, axis=0)

    """
    # Calculate Scale Factor
    if not skip_scale:
      self.I = np.ones((1,1))
      calc_coeffs = self.calculate_coeffs(self.atom_positions)
      if self.data_or_sim:
        if "scale" in self.data_params:
          if self.data_params["scale"] is None:
            self.I = np.sum(
                calc_coeffs/self.eval_data_coeffs_var*self.eval_data_coeffs, -1)\
                /np.sum(calc_coeffs/self.eval_data_coeffs_var*calc_coeffs, -1)
            self.I_std = np.sqrt(1.\
                /np.sum(calc_coeffs**2/self.eval_data_coeffs_var, -1))
          else:
            self.I = self.data_params["scale"]
        else:
          print("WARNING: Setting I based upon the first fitting entry. Highly discouraged")
          #print(calc_coeffs.shape, self.eval_data_coeffs_var.shape, self.eval_data_coeffs.shape)
          #print(np.sum(calc_coeffs/self.eval_data_coeffs_var*self.eval_data_coeffs, -1).shape)
          #print(np.sum(calc_coeffs/self.eval_data_coeffs_var*calc_coeffs, -1).shape)
          self.I = np.sum(
                calc_coeffs/self.eval_data_coeffs_var*self.eval_data_coeffs, -1)\
              /np.sum(calc_coeffs/self.eval_data_coeffs_var*calc_coeffs, -1)
          self.I_std = np.sqrt(1.\
              /np.sum(calc_coeffs**2/self.eval_data_coeffs_var, -1))
          #TODO: remove this expand dims
          self.I = self.I[0]#np.expand_dims(self.I[0], -1)
      if len(np.array(self.I).shape) < 2:
        self.I = np.reshape(self.I, (1, 1))
      self.init_fit = self.I*calc_coeffs

    print("Using I:", self.I)
    if plot and self.plot_setup:
      plot_coeffs = self.I*calc_coeffs
      for i in range(plot_coeffs.shape[0]):
        plt.errorbar(self.eval_dom, self.eval_data_coeffs[i,:], np.sqrt(self.eval_data_coeffs_var[i,:]))
        plt.plot(self.eval_dom, plot_coeffs[i,:])
        fileName = os.path.join("plots", self.get_fileName(0))
        plt.savefig(fileName + "_scaleInit-{}-{}-{}.png".format(*self.data_LMK[i,:]))
        plt.close()
    """

    """
    self.data_Lcalc = np.reshape(self.data_LMK[:,0], (-1, 1, 1))
    self.data_Mcalc = np.reshape(self.data_LMK[:,1], (-1, 1, 1))
    self.data_Kcalc = np.reshape(self.data_LMK[:,2], (-1, 1, 1))
    self.calculate_coeffs(self.atom_positions)
    """
    return self.atom_positions


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
  """

  def get_fileName(self, N=None, folder_only=False, suffix=None):

    dtype = "data"
    if "simulate_data" in self.data_params or not self.data_or_sim:
      if self.data_params["simulate_data"]:
        dtype = "sim"

    lg_name = "lg"
    for l in np.sort(np.unique(self.data_params["fit_bases"][:,0])):
      lg_name += "-{}".format(int(l))

    folder = os.path.join(self.data_params["molecule"],
        self.data_params["experiment"], dtype, lg_name)
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
          if etype == "StoN":
            variance_scale = variance_scale[0]
          etype = etype.lower()

          fileName += "_error-{0}_scale-{1:.3g}".format(
              etype, variance_scale)

    if N is not None:
      fileName += "_n-{}_N-{}".format(self.data_params["Nnodes"], N)

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
      h5.create_dataset("Nnodes", data=self.data_params["Nnodes"])



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
        self.data_params["Nnodes"] = h5["Nnodes"][...]

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
          [[] for i in range(self.data_params["Nnodes"])],\
          [[] for i in range(self.data_params["Nnodes"])]


  def get_spherical_j_calc(self, n):

    if "jn_calc" not in self.data_params:
      self.data_params["jn_calc"] = 0

    if self.data_params["jn_calc"] == 0:
      def numpy_jn(x):
        return sp.special.spherical_jn(self.data_Lcalc, x) 

      return numpy_jn

    elif self.data_params["jn_calc"] == 1:
      if np.sum(n % 2) == 0:
        ind = 0
        lmk = self.data_LMK[:,0]
        keep_inds = np.ones(len(lmk)).astype(bool) 
        for i in np.arange(np.amax(lmk)//2+1):
          for j in np.arange(len(lmk)):
            if i*2 < lmk[j] and i*2 not in lmk:
              print("inserting", j, i*2)
              lmk = np.insert(lmk, j, i*2, axis=0)
              keep_inds = np.insert(keep_inds, j, False, axis=0)
      
        def calc_even_only(x):
          return spherical_j(x, lmk)[keep_inds]

        return calc_even_only

    elif self.data_params["jn_calc"] == 2:
      
      if np.sum(n % 2) == 0:
        N = (np.unique(n)/2).astype(int)
        N_max = int(np.amax(N))
        N_elements = 2*N_max
        counts = []
        for nn in np.unique(n):
          counts.append(np.sum(nn == n))

        # Indices of invx for sin (even) and cos (odd) terms
        even_inds = np.where(np.arange(N_elements+1)%2 == 0)[0]
        odd_inds = np.where(np.arange(N_elements+1)%2 == 1)[0]
        print("EVN", N_max, even_inds)

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

          print(x.shape)
          tt = time.time()
          invx = [1./x]
          invx.append(invx[0]/x)
          invx = np.array(invx)
          print("T1", time.time() - tt)

          # Calculate j0
          tt = time.time()
          #res.append((np.sin(x)*invx[0]))
          #j0_ = np.cos(x)*invx[0]
          res.append(np.sin(x))
          j0_ = np.sqrt(1-res[0]**2)*invx[0]
          neg_inds = np.where((x%(2*np.pi)>np.pi/2) & (x%(2*np.pi)<3*np.pi/2))
          j0_[neg_inds] *= -1
          res[0] = res[0]*invx[0]
          print("T2", time.time() - tt)

          # Calculate j2
          tt = time.time()
          #res.append(np.exp(np.log(scales[1][0][0] +  scales[1][0][1]*invx[1]) + np.log(res[0]))
          #    + np.exp(np.log(scales[1][1][0]*invx[0])+np.log(j0_)))
          res.append(((scales[1][0][0] +  scales[1][0][1]*invx[1])*res[0]
              + scales[1][1][0]*invx[0]*j0_).astype(np.double))
          print("T3", time.time() - tt)

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
            print("tt", time.time() - t)
          print("T4", time.time() - tt)

          tt = time.time()
          for i in np.arange(N_max+1):
            if x[0,0,0,0] == 0:
              if i == 0:
                res[-1][:,0,:,:] = 1.
              else:
                res[-1][:,0,:,:] = 0
          print("T5", time.time() - tt)

          tt = time.time()
          ic = 0
          for i in range(len(res)):
            if keep_inds[i]:
              res[i] = np.tile(res[i], (counts[ic],1,1,1,1))
              ic += 1
          for i in reversed(np.where(np.invert(keep_inds))[0]):
            del res[i]
          print("T6", time.time() - tt)

          tt = time.time()
          a = np.concatenate(res, 0)
          print("T7", time.time() - tt)
          print("DONE!!!", a.shape)
          return np.concatenate(res, 0)

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
      return calc_even_only



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


