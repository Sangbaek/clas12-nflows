import pickle
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
from utils.utilities import cartesian_converter, spherical_converter
from utils.utilities import make_model
from utils import make_histos
from utils import dataXZ

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
  def __init__(self, standard = False, feature_subset = "all", file = "data/pi0toepg.pkl", mode = "epg"):
    #use if already converted to cartesian
    #with open('data/pi0_cartesian_train.pkl', 'rb') as f:
       #x = np.array(pickle.load(f), dtype=np.float32)

    #Use if not already converted
    with open(file, 'rb') as f:
        xb = np.array(pickle.load(f), dtype=np.float32)
    '''
    data structure changed.
    epgg.pkl : all epgg events
    pi0.pkl : only pi0 events
     0 column : event
     1–16 column : x
     17: e sector
     18: g1 sector
     19: g2 sector
     20–35 column: z
    '''

    #x = xb[:, 1:17]
    #z = xb[:, 20:]
    if mode == "epgg":
        b = xb[:, 1:17]
        x = xb[:, 20:]
    else:
        b = xb[:, 1:13]
        x = xb[:, 13:]

    #x = spherical_converter(x, mode = mode)
    #z = spherical_converter(z, mode = mode)
    
    if feature_subset != "all": 
      x = x[:,feature_subset]
      b = b[:,feature_subset]

    # xwithoutPid = x

    # self.qt = self.quant_tran(x)

    #Commented out because currently ton using Quant trans.
    # df_x = pd.DataFrame(self.qt.transform(x)) #Don't know how to do this without first making it a DF
    # x_np = df_x.to_numpy() #And then converting back to numpy
    # self.x = torch.from_numpy(np.array(x_np))

    x = x - b
    self.xb = xb
    self.x = torch.from_numpy(np.array(x))
    # self.xwithoutPid = torch.from_numpy(np.array(xwithoutPid))
    self.b = torch.from_numpy(np.array(b))


    # if standard:
    #   self.standardize()

  # def quant_tran(self,x):
  #   gauss_scaler = QuantileTransformer(output_distribution='normal').fit(x)
  #   return gauss_scaler

  # def standardize(self):
  #   self.xMu = self.xwithoutPid.mean(0)
  #   self.xStd = self.xwithoutPid.std(0)
  #   self.zMu = self.zwithoutPid.mean(0)
  #   self.zStd = self.zwithoutPid.std(0)
  #   self.xwithoutPid = (self.xwithoutPid - self.xMu) / self.xStd
  #   self.zwithoutPid = (self.zwithoutPid - self.zMu) / self.zStd

  # def restore(self, data, type = "x"):
  #   mu = self.xMu
  #   std = self.xStd
  #   if type == "z":
  #     mu = self.zMu
  #     std = self.zStd
  #   return data * std + mu

  def sample(self, n):
    randint = np.random.randint( self.xb.shape[0], size =n)
    xb = self.xb[randint]
    x = self.x[randint]
    b = self.b[randint]
    # xwithoutPid = self.xwithoutPid[randint]
    # zwithoutPid = self.zwithoutPid[randint]
    # return {"xb":xb, "x": x, "z": z, "xwithoutPid": xwithoutPid, "zwithoutPid": zwithoutPid}
    # return {"xb":xb, "x": x,"z": z, "xwithoutPid": xwithoutPid}
    return {"xb":xb, "x": x,"b": b}
