import sys
import time
import ctypes
import pathlib

import numpy as np
import numpy.ctypeslib as npct
from matplotlib import pyplot as plt


arr1_d = npct.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')
arr2_d = npct.ndpointer(dtype=np.double, ndim=2, flags='C')
arr3_d = npct.ndpointer(dtype=np.double, ndim=3, flags='C')
arr4_d = npct.ndpointer(dtype=np.double, ndim=4, flags='C')
arr1_i = npct.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
libcd = npct.load_library("c_calc_extensions.so", "./cpp_extensions/lib/")


#########################
#####  spherical_j  #####
#########################

libcd.spherical_j.restype = None
libcd.spherical_j.argtypes = [
    arr1_d, ctypes.c_ulonglong,
    ctypes.c_int, arr1_d,
    ctypes.POINTER(None)
]

def spherical_j(x, lmk):
  """
  Calculate Spherical Bessel functions of the first kind using this method's
  C++ implementation.

      Parameters
      ----------
      x : N dimensional np.array of type float64 [d1, d2, ..., dN]
          The points to evaluate the spherical bessel functions on.
      lmk : 1D np.array of type int32 [lmk]
          The L quantum number needed to be calculated. If values appear
          multiple times subsequent entries will be copies of the first.

      Returns
      -------
      result : N dimensional np.array of type float64 [l_max/2+1, d1, ..., dN]
          The even spherical bessel functions from 0 to the maximum l in lmk.
  """

  n_max = int(np.amax(lmk)) # Largest order to calculate up to
  inp_shape = x.shape
  result = np.ones( # Pass results by reference
      (n_max//2+1)*np.prod(inp_shape),
      dtype=np.double, order="C")

  # Calculating the spherical bessel functions
  libcd.spherical_j(
      np.ascontiguousarray(np.reshape(x, -1).astype(np.double)),
      np.prod(inp_shape),
      n_max,
      result,
      None)
  result = np.reshape(result, [n_max//2+1] + list(inp_shape))

  return result


#############################
#####  calculate_c_cpp  #####
#############################

libcd.calculate_c.restype = None
libcd.calculate_c.argtypes = [
    arr4_d, arr1_i,
    arr1_i, ctypes.c_int,
    arr1_d, arr1_d, arr4_d, arr4_d,
    arr3_d, arr2_d, arr3_d
]

def calculate_c_cpp(x, lmk, c_prefactor, Ylk, scat_amps, weights):
  """
  Calculates the C coefficients using the C++ implementation of the Spherical
  Bessel functions of the first kind

      Parameters
      ----------
      x : 4D np.array of type float64 [dists,q,batch,ensemble]
          The batched arguments for the spherical bessel functions (q*deltaR) 
          for each molecule in the ensemble.
      lmk : 1D np.array of type int32 [lmk]
          The L quantum number for each C coefficient indicating the order of
          anisotropy and the order of the spherical bessel function.
      c_prefactor : 1D np.array of type float64 [lmk]
          Normalization factors for each L contribution for the C coefficients.
      Ylk : 4D np.array of type complex float64 [lmk,dists,batch,ensemble]
          The spherical harmonic terms of the C coefficient calculation
      scat_amps : 3D np.array of type float64 [1,dists,q].
          Atomic scattering amplitude factor for the pair-wise distances.
      weights : 2D np.array of type float64 [batch,ensemble]
          The weights of each molecule in the ensemble.

      Returns
      -------
      result : 3D np.array of type float64 [lmk,q,batch]
          The calculated C coefficients. 
  """

  (N_dists, N_qbins, N_batch, N_mols) = x.shape  # Get and label dimensions
  result = np.zeros((len(lmk), N_batch, N_qbins), # Pass results by reference
      order="C", dtype=np.double)

  # Calculate the C coefficients
  libcd.calculate_c(
      np.ascontiguousarray(np.transpose(x.astype(np.double), (2,3,0,1))),
      np.array(x.shape, dtype=np.int32, order="C"),
      lmk.astype(np.int32),
      len(lmk),
      np.ascontiguousarray(np.real(c_prefactor).astype(np.double)),
      np.ascontiguousarray(np.imag(c_prefactor).astype(np.double)),
      np.ascontiguousarray(
          np.transpose(np.real(Ylk), (0,2,3,1)).astype(np.double)), 
      np.ascontiguousarray(
          np.transpose(np.imag(Ylk), (0,2,3,1)).astype(np.double)),
      np.ascontiguousarray(scat_amps.astype(np.double)),
      np.ascontiguousarray(weights.astype(np.double)),
      result)
  result = np.transpose(result, (0,2,1)) # shape [lmk, q, batch]

  return result
