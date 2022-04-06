import sys
import ctypes
import pathlib
import numpy as np
import numpy.ctypeslib as npct
from matplotlib import pyplot as plt

arr_d = npct.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')
arr_i = npct.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
libcd = npct.load_library("spherical_j_cpp.so", "./cpp_extensions/lib/")

libcd.spherical_j.restype = None
libcd.spherical_j.argtypes = [
    arr_d, arr_i, ctypes.c_int,
    arr_i, ctypes.c_int, arr_d]

def spherical_j(x, lmk):
  """
  Calculate Spherical Bessel functions of the first kind using by importing
  the C++ implementation using numpy.ctypeslib and ctypes

  x : 4D np.array of type float64 [dists,q,batch,ensemble]
      The batched arguments for the spherical bessel functions (q*dists) for
      each molecule in the ensemble
  lmk : 1D np.array of type int32 [lmk]
      The L quantum number needed to be calculated. If values appear multiple
      times subsequent entries will be copies of the first.
  """

  inp_shape = x.shape
  result = np.zeros(np.prod(inp_shape)*len(lmk), dtype=np.double)
  libcd.spherical_j(
      np.reshape(x, (-1)).astype(np.double),
      np.array(inp_shape).astype(np.int32), len(inp_shape),
      lmk.astype(np.int32), len(lmk), result)
  result = np.reshape(result, [len(lmk)] + list(inp_shape))
  return result


libcd.calc_coeffs_helper.restype = None
libcd.calc_coeffs_helper.argtypes = [
    arr_d, arr_i, ctypes.c_int,
    arr_i, ctypes.c_int,
    arr_d, arr_d, arr_d, arr_d,
    arr_d, arr_d, arr_d]

def calc_coeffs_cpp_helper(x, lmk, C, Y, scat_amps, weights):
  """
  Calculates the C coefficients using the C++ implementation of the Spherical
  Bessel function of the first kind

  x : 4D np.array of type float64 [dists,q,batch,ensemble]
      The batched arguments for the spherical bessel functions (q*dists) for
      each molecule in the ensemble
  lmk : 1D np.array of type int32 [lmk]
      The L quantum number needed to be calculated. If values appear multiple
      times subsequent entries will be copies of the first.
  C : 1D np.array of type float64 [lmk]
      Normalization factors for each L contribution for the C coefficients
  Y : 4D np.array of type complex float64 [lmk,dists,batch,ensemble]
      The spherical harmonic terms of the C coefficient calculation
  scat_amps : 3D np.array of type float64 [1,dists,ensemble]
      The atomic scattering amplitudes for the two atoms making each distance
  weights : 2D np.array of type float64 [batch,ensemble]
      The weights of each molecule in the ensemble
  """

  inp_shape = x.shape
  result = np.zeros(inp_shape[1]*inp_shape[2]*len(lmk), dtype=np.double)
  libcd.calc_coeffs_helper(
      np.reshape(x, (-1)).astype(np.double),
      np.array(inp_shape, dtype=np.int32), len(inp_shape),
      lmk.astype(np.int32), len(lmk),
      np.real(C).astype(np.double), np.imag(C).astype(np.double),
      np.reshape(np.real(Y), (-1)).astype(np.double), 
      np.reshape(np.imag(Y), (-1)).astype(np.double),
      np.reshape(scat_amps, (-1)).astype(np.double),
      np.reshape(weights, (-1)).astype(np.double), result)
  result = np.reshape(result, [len(lmk), inp_shape[1], inp_shape[2]])
  return result
