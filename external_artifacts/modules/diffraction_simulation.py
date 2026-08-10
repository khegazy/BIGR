import sys, os, glob
import pickle as pl
import numpy as np
import scipy as sp
import h5py
from copy import copy as copy
import matplotlib.pyplot as plt
#import spherical_functions as sf
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import factorial, jacobi
from multiprocessing import Pool
from functools import partial


def get_wignerD_comparison(phi_ea, theta_ea, chi_ea, l, m):
  lmm = np.ones((2*l+1, 3))*l
  lmm[:,1] -= np.arange(2*l+1)
  lmm[:,2] = m
  lmm = lmm.astype(int)

  D_ME = sf.Wigner_D_element(phi_ea, theta_ea, chi_ea, lmm)
  return D_ME, lmm


def small_factorial(start, end):
  start = np.max(start, 1)
  return np.prod(np.arange(start, end+1))

def nCk(n, k):
  return small_factorial(k,n)/small_factorial(1, n-k+1)

def nCk_np(n, k):
  return factorial(n)/(factorial(k)*factorial(n-k))

def Ylm_calc(m, l, azim, polar):
  return sp.special.sph_harm_y(l, m, polar, azim).astype(np.complex64)
  #return np.abs(-1)**np.abs(m)*sp.special.sph_harm(m, l, azim, polar, dtype=np.complex64)

def get_wignerD_3d(phi_ea, theta_ea, chi_ea, l, m, k):
  # Return array of D matrices with L, M, K as l, [-l, l], m
  # Calculation based off of below derivation
  # https://en.wikipedia.org/wiki/Wigner_D-matrix#Wigner_(small)_d-matrix

  m = np.expand_dims(m, 0)
  phi_ea    = np.expand_dims(phi_ea, axis=-1)
  theta_ea  = np.expand_dims(theta_ea, axis=-1)
  chi_ea    = np.expand_dims(chi_ea, axis=-1)

  k_array = np.array([l+k, l-k, l, l], dtype=int)
  k_matrix = np.tile(np.expand_dims(k_array, -1), m.shape[1])

  k_matrix[2,:] += m[0]
  k_matrix[3,:] -= m[0]

  a_array = np.array([-1*k, k, k, -1*k])
  a_matrix = np.tile(np.expand_dims(a_array, -1), m.shape[1])
  a_matrix += np.array([[1,-1,-1,1]]).transpose()*m

  k_inds = np.argmin(k_matrix, axis=0)
  k_, a = np.zeros_like(m), np.zeros_like(m)
  ii = np.arange(k_inds.shape[0])
  k_[0,:] = k_matrix[k_inds[ii],ii]
  a[0,:] = a_matrix[k_inds[ii],ii]
  b = 2*l - 2*k_ - a
  lmbd = (m - k)*np.maximum(0, (1-(k_inds%3)))
  #lmbd = np.maximum(0, (m - k))
  #print(lmbd, (m-k), (1-(k_inds%3)), k_inds)
  #print("ll",lmbd)


  # Calculate little d
  d_coeff = (-1)**lmbd*np.sqrt(nCk_np(2*l-k_, k_+a))/np.sqrt(nCk_np(k_+b, b))\
      *np.sin(theta_ea/2)**a*np.cos(theta_ea/2)**b
  d = np.zeros_like(d_coeff)
  cos_theta = np.cos(theta_ea)
  jac = []
  for im in range(m.shape[-1]):
    p = jacobi(k_[0,im], a[0,im], b[0,im])
    jac.append(np.polyval(p, cos_theta)[:,0])
    d[:,im] = d_coeff[:,im]*np.polyval(p, cos_theta)[:,0]

  #print("d", d)
  #print("jac", jac)
  #print("l", lmbd)
  #print("mmmmmmmmmmm", m, (-1j))
  #print("phi", phi_ea)
  #print("res",np.exp((-1j)*m*phi_ea))
  # Calculate D matrix element
  D_ME = np.exp((-1j)*m*phi_ea)*d*np.exp((-1j)*k*chi_ea)
  #D_ME = d*(np.cos(-1*m*phi_ea) + np.cos(-1*k*chi_ea)\
  #    + 1j*(np.sin(-1*m*phi_ea) + np.sin(-1*k*chi_ea)))
  #print(-1*m*phi_ea)

  return D_ME.transpose().astype(np.complex64)

def rotate_Ylm_3d(phi_ea, theta_ea, chi_ea, l, m, polar, azim):
  m_sum = np.ones(2*l+1)*l - np.arange(2*l+1)
  m_sum = m_sum.astype(np.int16)
  #m_sum = np.ones((1, 2*l+1), dtype=np.int16)
  #m_sum[0,:] = np.ones(2*l+1)*l - np.arange(2*l+1, dtype=np.int16)
  D_ME = get_wignerD_3d(phi_ea, theta_ea, chi_ea, l, m_sum, m)

  """
  print("D SIZE", D_ME.shape)
  print(D_ME)
  """
  Ylms = Ylm_calc(
    np.expand_dims(np.expand_dims(m_sum, -1), -1),
    np.ones((m_sum.shape[0], 1, 1))*l,
    np.expand_dims(azim,0),        
    np.expand_dims(polar,0))

  return np.einsum('ia,ibc->abc', D_ME, Ylms)
  #return np.sum(
  #    np.expand_dims(np.expand_dims(D_ME, -1), -1)\
  #    *np.expand_dims(Ylms, 1), axis=0)


def calc_dists(R):
  r     = np.expand_dims(R, 0) - np.expand_dims(R, 1)
  dR    = np.sqrt(np.sum(r**2, axis=-1))
  theta = np.arccos(r[:,:,2]/(dR + 1e-20))
  phi   = np.arctan2(r[:,:,1], r[:,:,0])
  mask = np.expand_dims((dR[:,:] > 0) , -1)

  return mask*np.concatenate([np.expand_dims(dR,-1),\
    np.expand_dims(theta, -1),\
    np.expand_dims(phi, -1)], axis=-1)



def atomic_diffraction_calculation(
    q_map, atom_types, scat_amps, detector_dist=1):

  if len(q_map.shape) == 3:
    q_map = q_map[:,:,0]

  diffraction = np.zeros((int(q_map.shape[0]), int(q_map.shape[1])))
  for imol in range(len(atom_types)):
    for atm in atom_types[imol]:
      diffraction += (1/detector_dist**2)\
          *scat_amps[atm](q_map)**2
  
  return diffraction






################################
#####  Numeric Simulation  #####
################################

def diffraction_calculation_numeric(
    molecule, itm, times, LMK, bases,
    detector_shape, detector_dist, q_map_polar,
    atom_types, atom_distances_polar,
    smearing_ea, smearing_jacobian, scat_amps, logging=None):

  smearing_weights = []
  for lmk in LMK:
    smearing_weights.append(
        ((2*lmk[0] + 1)/(8*np.pi**2))*get_wignerD_3d(
          smearing_ea[:,0],
          smearing_ea[:,1],
          smearing_ea[:,2],
          lmk[0], np.array([lmk[1]]), lmk[2]))
    #smearing_weights[-1] /= np.sqrt(4*np.pi/(2*lmk[0] + 1))

 
  """
  if molecule == "N2O":
    print("NORMING")
    smearing_weights = 2*np.pi**2*np.conj(smearing_weights)\
        /(4*np.pi/(2*lmk[0] + 1)
  elif molecule == "NO2":
    smearing_weights /= np.power(np.sqrt(4*np.pi/(2*lmk[0] + 1)), 1.5)
  """

  norm = np.sqrt(4*np.pi/(2*LMK[:,0] + 1))
  #smearing_weights = np.real(np.sum(
  smearing_weights = np.sum(
      np.expand_dims(norm*bases[:,itm], axis=-1)\
        *np.concatenate(smearing_weights, axis=0),
      axis=0)#.astype(np.float32)

  smearing_weights *= smearing_jacobian
 
  """
  def diffraction_calculation_numerical(
    detector_shape, detector_dist, q_map_polar,
    atom_types, atom_distances_polar,
    smearing_ea, smearing_weights,
    scat_amps, logging=None):
  """

  n = 4000
  N_points = smearing_weights.shape[0]
  """
  plt.plot(np.cos(smearing_ea[:,1]), smearing_weights)
  plt.savefig("testw.png")
  plt.close()
  """
  smearing_weights = np.expand_dims(np.expand_dims(smearing_weights, -1), -1)
  diffraction = np.zeros((int(detector_shape[0]), int(detector_shape[1])),
      dtype=complex)

  #####  Loop over smearing points  #####
  for imol in range(len(atom_distances_polar)):
    for igeo in range(len(atom_distances_polar[imol])):
      lc,hc = 0,np.min([n, N_points])
      while hc < N_points or lc == 0:
        if logging is not None:
          logging.info("Evaluating range {} - {} of {}".format(lc, hc, N_points))
        else:
          print("Evaluating range {} - {} of {}".format(lc, hc, N_points))

        ###  Loop over all pairwise distances  #####
        for ir, r_diff in enumerate(atom_distances_polar[imol][igeo]):
          """
          rr = 40
          cent = int(detector_shape[1]/2)
          print(cent, smearing_ea[lc:hc,1].shape)
          irr = (rr*np.cos(smearing_ea[lc:hc,1])).astype(int) + cent
          icc = (rr*np.cos(smearing_ea[lc:hc,0])).astype(int) + cent
          print("WEIGHT SH", smearing_weights.shape)
          diffraction[irr,icc] += smearing_weights[lc:hc,0,0]
          continue
          """

          # Rotate Y10 so it points in direction of r in lf
          #D_ME_, lmm = get_wignerD(r_diff[2], r_diff[1], 0, 1, 0)
          l = 1
          m_sum = np.ones(3) - np.flip(np.arange(3))
          m_sum = m_sum.astype(np.int16)
          D_ME = get_wignerD_3d(
              np.array([r_diff[2]]), np.array([r_diff[1]]), np.zeros(1),
              1, m_sum, 0)
          #    np.array([r_diff[2]]), np.array([r_diff[1]]), np.zeros(1), 1, 0)
          D_ME = D_ME[:,0]

          q_dot_r = np.zeros(
              (hc - lc, q_map_polar.shape[0], q_map_polar.shape[1]),
              dtype=np.complex64)
          for im, m in enumerate(m_sum):
            if logging is not None:
              logging.info("\tir / m: {} / {}".format(ir, m))
            else:
              print("\tir / m: {} / {}".format(ir, m))

            if np.abs(D_ME[im]) <= 1e-10:
              print("skipping")
              continue

            """
            rr,cc = 3,4
            Nb = 10
            tst = rotate_Ylm_3d(
              smearing_ea[:Nb,0], smearing_ea[:Nb,1], smearing_ea[:Nb,2],\
                  l, m, q_map_polar[:,:,1], q_map_polar[:,:,2])
            print("VAR", np.sum(np.abs(tst - np.mean(tst,axis=0,keepdims=True))**2))
            """

            q_dot_r +=  D_ME[im]*(q_map_polar[:,:,0]*r_diff[0]/(0.5*np.sqrt(3/np.pi)))\
                  *rotate_Ylm_3d(
                      smearing_ea[lc:hc,0], smearing_ea[lc:hc,1], smearing_ea[lc:hc,2],\
                      l, m, q_map_polar[:,:,1], q_map_polar[:,:,2])

          #print("QDR", np.amax(np.real(q_dot_r)), np.amax(np.imag(q_dot_r)))
          diffraction +=\
              scat_amps[atom_types[imol][ir][0]](q_map_polar[:,:,0])\
              *scat_amps[atom_types[imol][ir][1]](q_map_polar[:,:,0])/(detector_dist**2)\
              *np.sum(smearing_weights[lc:hc]*np.cos(np.real(q_dot_r)), axis=0)
          #diffraction +=\
          #    scat_amps[atom_types[ir][0]](q_map_polar[:,:,0])\
          #    *scat_amps[atom_types[ir][1]](q_map_polar[:,:,0])/(detector_dist**2)\
          #    *np.real(np.sum(smearing_weights[lc:hc]*np.cos(q_dot_r), axis=0))
          #    #*np.sum(smearing_weights[lc:hc]*np.cos(np.real(q_dot_r)), axis=0)
        lc = hc
        hc = np.min([hc+n, N_points])

        """
          diffraction +=\
              scat_amps[atom_types[ir][0]](q_map_polar[:,:,0])\
              *scat_amps[atom_types[ir][1]](q_map_polar[:,:,0])/(detector_dist**2)\
              *np.sum(smearing_weights*np.cos(
                np.real(D_ME[im]*(q_map_polar[:,:,0]*r_diff[0]/(0.5*np.sqrt(3/np.pi)))\
                *rotate_Ylm_3d(smearing_ea[:,0], smearing_ea[:,1], smearing_ea[:,2],\
                  l, m, q_map_polar[:,:,1], q_map_polar[:,:,2]))),
                axis=0)
        """
        """
          q_dot_r_lf = np.real(D_ME[im]*(q_map_polar[:,:,0]*r_diff[0]/(0.5*np.sqrt(3/np.pi)))\
              *rotate_Ylm_3d(smearing_ea[:,0], smearing_ea[:,1], smearing_ea[:,2],\
                l, m, q_map_polar[:,:,1], q_map_polar[:,:,2]))
          smearing = np.sum(smearing_weights*np.cos(q_dot_r_lf), axis=0)
          diffraction += smearing*scat_amps[atom_types[ir][0]](q_map_polar[:,:,0])\
              *scat_amps[atom_types[ir][1]](q_map_polar[:,:,0])/(detector_dist**2)
        """
  
  return np.real(diffraction), diffraction



#################################
#####  Analytic Simulation  #####
#################################


def molecular_diffraction_calculation(
    LMK, LMK_weights,
    R, atom_types, scat_amps,
    q_map, detector_dist=1, freq=None):
  """
  Analytically calculates the molecular diffraction component for an
  anisotropic ensemble of molecules.

  Parameters
  ----------
  LMK : np.array
      Anisotropy quantum numbers. Shape [Nl,3].
  LMK_weights : np.array
      Weight of each anisotropy element. Shape [time, Nl] or [Nl].
  R : np.array
      Cartesian coordinates of the molecule. Shape [molecules, geometries, atoms, 3].
  atom_types : list
      List of strings denoting the element type. Shape [molecules, atoms].
  scat_amps : dict
      Dictionary of interp1d functions with key elements in `atom_types`.
  q_map : np.array
      2d map of either q, theta_lf, phi_lf for each pixel or just q. If only q magnitude is given then `freq` must be specified. Shape [Npix, Npix, 3] or [Npix, Npix].
  detector_dist : np.array, optional
      The distance between the interaction point and each pixel. This must be specified for electron diffraction only.
  freq : float, optional
      Either the x-ray or electron deBroglie wavelength in inverse units of `q_map`.

  Returns
  -------
  np.array
      Molecular diffraction pattern with shape [Npix, Npix] or [time, Npix, Npix].
  """

  ###  Setup Parameters  ###
  L = np.reshape(LMK[:,0], (-1, 1, 1, 1))
  M = np.reshape(LMK[:,1], (-1, 1, 1, 1))
  K = np.reshape(LMK[:,2], (-1, 1, 1, 1))
  if len(LMK_weights.shape) == 2:
    dfr_shape = (LMK_weights.shape[0], q_map.shape[0], q_map.shape[1])
  else:
    dfr_shape = (q_map.shape[0], q_map.shape[1])
  diffraction = np.zeros(dfr_shape, dtype=np.complex64)

  # Scattering Angles
  if len(q_map.shape) == 3:
    theta_lf = q_map[:,:,1]
    phi_lf = q_map[:,:,2]
    q_map = q_map[:,:,0]
  else:
    if freq is None:
      raise ValueError("Calculate Diffraction: "\
          +"Must specify freq if lab frame angles are not given in q_map\n")

    qx,qy = np.meshgrid(np.arange(q_map.shape[0]), np.arange(q_map.shape[1]))
    qx -= q_map.shape[0]//2
    qy -= q_map.shape[0]//2
    theta_d   = np.arctan2(qx, qy)
    _theta_d = copy(theta_d)
    _theta_d[theta_d == 0] = 1
    _theta_d[theta_d == np.pi] = 1
    alpha     = 2*np.arcsin(q_map[0]*freq/(4*np.pi)) + np.pi/2
    theta_lf  = np.arccos(np.sin(alpha)*np.cos(theta_d))
    phi_lf    = np.arctan(np.cos(alpha)/(np.sin(alpha)*np.sin(_theta_d)))
    phi_lf[theta_d == 0] = -1*np.pi
    phi_lf[theta_d == np.pi] = -1*np.pi
    theta_d[:,:q_map.shape[0]//2] += 2*np.pi# - theta_d[:,:q_map.shape[0]//2]
    phi_lf[:,:q_map.shape[0]//2] += np.pi
  """
  qx,qy = np.meshgrid(np.arange(q_map.shape[0]+1), np.arange(q_map.shape[1]+1))
  plt.pcolormesh(qx,qy, theta_d)
  plt.colorbar()
  plt.savefig("thetad.png")
  plt.close()
  plt.pcolormesh(qx,qy, theta_lf)
  plt.colorbar()
  plt.savefig("thetalf.png")
  plt.close()
  plt.pcolormesh(qx,qy, phi_lf)
  plt.colorbar()
  plt.savefig("philf.png")
  plt.close()
  """

  for imol in range(len(atom_types)):
    for igeo in range(len(R[imol])):
      all_dists = calc_dists(R[imol][igeo])
      dist_inds1 = []
      dist_inds2 = []
      dist_scat_amps = []
      for i, a1 in enumerate(atom_types[imol]):
        for j, a2 in enumerate(atom_types[imol]):
          if i == j:
            continue
          dist_inds1.append(i)
          dist_inds2.append(j)
          dist_scat_amps.append(
              scat_amps[a1](q_map)*scat_amps[a2](q_map))
      dist_scat_amps = np.array(dist_scat_amps)
      
      # Aligning distances with scattering amps
      dist_inds = (np.array(dist_inds1), np.array(dist_inds2))
      dists = all_dists[dist_inds]

      C = ((-1)**K)*1j**L
      C *= 32*np.pi**3/(2*L + 1)
      J = sp.special.spherical_jn(L,
          q_map[:,:]*np.expand_dims(np.expand_dims(dists[:,0], axis=-1), -1))
      # sph_harm_y(n, m, polar, azim) replaces sph_harm(m, n, azim, polar):
      # dists[:,1] is the polar and dists[:,2] the azimuthal angle.
      Y = sp.special.sph_harm_y(L, -1*K,
          np.expand_dims(np.expand_dims(dists[:,1], axis=-1), axis=-1),
          np.expand_dims(np.expand_dims(dists[:,2], axis=-1), axis=-1))


      _diffraction = dist_scat_amps*C*J*Y
      diffraction += np.sum(np.sum(
          np.expand_dims(LMK_weights, (-1, -2, -3))*_diffraction\
          *sp.special.sph_harm_y(L, -1*M,
              np.expand_dims(theta_lf, axis=0),
              np.expand_dims(phi_lf, axis=0)),
              axis=-3), axis=-3)

  diffraction /= detector_dist

  return np.real(diffraction), diffraction


def diffraction_calculation(
    LMK, LMK_weights,
    R, atom_types, scat_amps,
    q_map, detector_dist=1, freq=None):

  atm_diffraction = atomic_diffraction_calculation(
      q_map, atom_types, scat_amps, detector_dist)

  mol_diffraction = molecular_diffraction_calculation(
    LMK, LMK_weights,
    R, atom_types, scat_amps,
    q_map, detector_dist=detector_dist, freq=freq)

  return atm_diffraction, mol_diffraction
