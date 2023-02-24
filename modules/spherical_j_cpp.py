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
libcd = npct.load_library("spherical_j.so", "./cpp_extensions/lib/")

libcd.spherical_j.restype = None
libcd.spherical_j.argtypes = [
    arr1_d, ctypes.c_ulonglong,
    ctypes.c_int, arr1_d,
    ctypes.c_int, ctypes.POINTER(None)]

def spherical_j(x, lmk, N_qbins=-1):
  """
  Calculate Spherical Bessel functions of the first kind using this method's
  C++ implementation.

      Parameters
      ----------
      x : N dimensional np.array of type float64 [d1, d2, ..., q]
          The pointes to evaluate the spherical bessel functinos on, where the
          last dimension is the number of q bins.
      lmk : 1D np.array of type int32 [lmk]
          The L quantum number needed to be calculated. If values appear
          multiple times subsequent entries will be copies of the first.

      Returns
      -------
      result : N dimensional np.array of type float64 [l_max/2+1, d1, ..., q]
          The even spherical spherical bessel functions from 0 to the maximum
          l in lmk.
  """

  n_max = int(np.amax(lmk))
  inp_shape = x.shape
  result = np.ones(
      (n_max//2+1)*np.prod(inp_shape),
      dtype=np.double, order="C")
  libcd.spherical_j(
      np.ascontiguousarray(np.reshape(x, -1).astype(np.double)),
      np.prod(inp_shape),
      n_max,
      result,
      inp_shape[-1],
      None)
  result = np.reshape(result, [n_max//2+1] + list(inp_shape))
  return result


libcd.calc_c_coeffs.restype = None
libcd.calc_c_coeffs.argtypes = [
    arr4_d, arr1_i,
    arr1_i, ctypes.c_int,
    arr1_d, arr1_d, arr4_d, arr4_d,
    arr3_d, arr2_d, arr3_d]

def calc_c_coeffs_cpp(x, lmk, C_prefactor, Y, scat_amps, weights):
  """
  Calculates the C coefficients using the C++ implementation of the Spherical
  Bessel function of the first kind

      Parameters
      ----------
      x : 4D np.array of type float64 [dists,q,batch,ensemble]
          The batched arguments for the spherical bessel functions (q*dists) for
          each molecule in the ensemble
      lmk : 1D np.array of type int32 [lmk]
          The L quantum number needed to be calculated. If values appear multiple
          times subsequent entries will be copies of the first.
      C_prefactor : 1D np.array of type float64 [lmk]
          Normalization factors for each L contribution for the C coefficients
      Y : 4D np.array of type complex float64 [lmk,dists,batch,ensemble]
          The spherical harmonic terms of the C coefficient calculation
      scat_amps : 3D np.array of type float64 [1,dists,q]
          The atomic scattering amplitudes for the two atoms making each distance
      weights : 2D np.array of type float64 [batch,ensemble]
          The weights of each molecule in the ensemble

      Returns
      -------
      result : 3D np.array of type float64 [
  """


  (N_dists, N_qbins, N_batch, N_mols) = x.shape
  result = np.zeros((len(lmk), N_batch, N_qbins),
      order="C", dtype=np.double)
  tic = time.time()
  print("N", lmk)
  print("shape test")
  print(x.shape)
  print(Y.shape)
  print(scat_amps.shape)
  print(weights.shape, np.sum(weights))
  print(weights)
  print(weights[weights>1e-3])
  plt.hist(weights[0], bins=100)
  plt.savefig("wtf_weights.png")
  libcd.calc_c_coeffs(
      np.ascontiguousarray(np.transpose(x.astype(np.double), (2,3,0,1))),
      np.array(x.shape, dtype=np.int32, order="C"),
      lmk.astype(np.int32),
      len(lmk),
      np.ascontiguousarray(np.real(C_prefactor).astype(np.double)),
      np.ascontiguousarray(np.imag(C_prefactor).astype(np.double)),
      np.ascontiguousarray(
          np.transpose(np.real(Y), (0,2,3,1)).astype(np.double)), 
      np.ascontiguousarray(
          np.transpose(np.imag(Y), (0,2,3,1)).astype(np.double)),
      np.ascontiguousarray(scat_amps.astype(np.double)),
      np.ascontiguousarray(weights.astype(np.double)),
      result)
  print("Finished C calc C++", time.time() - tic)
  print("test shape", x.shape)
  #result = np.reshape(result, [len(lmk), N_batch, N_qbins])
  result = np.transpose(result, (0,2,1)) # shape [lmk, q, batch]
  print(result[0,:,0])
  print("end")
  return result
