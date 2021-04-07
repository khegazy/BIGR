import numpy as np
import csv
import os, glob, sys, shutil
import time
import h5py
import pickle as pl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy as copy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d



def get_NO2_ADMs(
    data_LMK, kwargs={},
    subtract_mean=False, normalize=False):

  bases_sub_dir = "A/temp-{}K".format(int(kwargs["temperature"]))
  folders = [kwargs["folder"], "UED", "NO2", bases_sub_dir]
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
  print("files", files)
  print(glob.glob(files))
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
      print(times_interp.shape, basis.shape)
      basis_interp = interp1d(times_inp, basis, kind='cubic')
      basis = basis_interp(times_interp)
      times = times_interp

      print("SHAPE", basis.shape, times.shape)
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

  LMK      = np.array(LMK)
  allBases = np.array(basisList)
  allNorms = np.array(normsList)

  fit_bases = []
  for lmk_ in data_LMK:
    print(lmk_)
    lInds = LMK[:,0] == lmk_[0]
    mInds = LMK[:,2] == lmk_[2]

    fit_bases.append(allBases[lInds*mInds])
    print(allBases[lInds*mInds].shape)





  return np.concatenate(fit_bases, axis=0)

