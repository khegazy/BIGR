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
libcd.spherical_j.argtypes = [arr_d, ctypes.c_int, arr_i, ctypes.c_int, arr_d]

def spherical_j(x, lmk):
  inp_shape = x.shape
  result = np.zeros(np.prod(inp_shape)*len(lmk), dtype=np.double)
  libcd.spherical_j(
      np.reshape(x, (-1)), np.prod(inp_shape),
      lmk.astype(np.int32), len(lmk), result)
  result = np.reshape(result, [len(lmk)] + list(inp_shape))
  return result
