import pickle5 as pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('pdf')
import itertools
import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch import optim
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, QuantileTransformer

from utils.utilities import meter
from utils.utilities import cartesian_converter
from utils.utilities import make_model
from utils import make_histos
from utils import dataXZ

sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')
from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation



#Create data class
class dataXZ:
  """
  read the data stored in pickle format
  the converting routine is at https://github.com/6862-2021SP-team3/hipo2pickle
  """
  def __init__(self, standard = False, feature_subset = "all"):
    with open('data/epgg_cartesian_train.pkl', 'rb') as f:
        #Since we already converted to cartesian, the below 2 lines are not needed
        #xz = np.array(pickle.load(f), dtype=np.float64)
        #x = cartesian_converter(xz)

        xz = np.array(pickle.load(f), dtype=np.float64)
        if feature_subset == "all": 
          feature_subset = [i for i in range(16)]
        xfeature_subset = [i+1 for i in feature_subset]
        zfeature_subset = [i+20 for i in feature_subset]
        xz = xz[:, xfeature_subset + zfeature_subset]

        self.qt = self.quant_tran(xz)

        df_xz = pd.DataFrame(self.qt.transform(xz)) #Don't know how to do this without first making it a DF
        xz_np = df_xz.to_numpy() #And then converting back to numpy
        x_np = xz_np[:, xfeature_subset]
        zfeature_subset2 = [i+len(feature_subset) for i in feature_subset]
        z_np = xz_np[:, zfeature_subset2]
        self.x = torch.from_numpy(np.array(x_np))
        self.z = torch.from_numpy(np.array(z_np))


    if standard:
      self.standardize()

  def quant_tran(self,x):
    gauss_scaler = QuantileTransformer(output_distribution='normal').fit(x)
    return gauss_scaler

  def standardize(self):
    self.xMu = self.xwithoutPid.mean(0)
    self.xStd = self.xwithoutPid.std(0)
    self.zMu = self.zwithoutPid.mean(0)
    self.zStd = self.zwithoutPid.std(0)
    self.xwithoutPid = (self.xwithoutPid - self.xMu) / self.xStd
    self.zwithoutPid = (self.zwithoutPid - self.zMu) / self.zStd

  def restore(self, data, type = "x"):
    mu = self.xMu
    std = self.xStd
    if type == "z":
      mu = self.zMu
      std = self.zStd
    return data * std + mu

  def sample(self, n):
        randint = np.random.randint( self.x.shape[0], size =n)
        x = self.x[randint]
        z = self.z[randint]
        return {"x": x, "z": z}

