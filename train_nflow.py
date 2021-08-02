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
from utils.utilities import cartesian_converter
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

# Define device to be used
dev = "cuda" if torch.cuda.is_available() else "cpu"
#dev = "cpu"
device = torch.device(dev)
print(dev)

#Define hyperparameters
#The number of featuers is just the length of the feature subset, or 16 if "all"
#feature_subset = [1,2,3,5,6,7,9,10,11,13,14,15] #Only 3 momenta (assuming PID is known)
feature_subset = [1,2,3] #Just electron features
part = "elec"
#feature_subset = [5,6,7] #Just proton features
#part = "prot"
#feature_subset = [9,10,11] #Just photon features
#part = "phot"
#feature_subset = "all" #All 16 features

#These are parameters for the Normalized Flow model
num_layers = 6
num_hidden_features = 80

#These are training parameters
num_epoch = 10000
training_sample_size = 128


if feature_subset == "all":
  num_features = 16
else:
  num_features = len(feature_subset)


#read the data, with the defined data class
xb = dataXZ.dataXZ(feature_subset=feature_subset, file = "data/train.pkl", mode="epg")
print("done with reading data")
print(xb.reco)
print(xb.truth)
#construct an nflow model
flow, optimizer = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow.parameters()))



start = datetime.now()
start_time = start.strftime("%H:%M:%S")
print("Start Time =", start_time)
losses = []
for i in range(num_epoch):
    sampleDict = xb.sample(training_sample_size)
    reco_train = sampleDict["reco"][:, 0:num_features].to(device)
    truth_train = sampleDict["truth"][:, 0:num_features].to(device)

    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=reco_train,context=truth_train).mean()

    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if ((i+1)%10) == 0:
      now = datetime.now()
      elapsedTime = (now - start )
      print("On step {} - loss {:.2f}, Current Time = {}".format(i,loss.item(),now.strftime("%H:%M:%S")))
      print("Elapsed time is {}".format(elapsedTime))
      print("Rate is {} seconds per epoch".format(elapsedTime/i))
      print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(num_epoch+1-i)))
      if ((i+1)%100) == 0:
        torch.save(flow.state_dict(), "models/Cond/3features/TM-UMNN_"+part+"_{}_{}_{}_{}_{}_{:.2f}.pt".format(num_features,
          num_layers,num_hidden_features,training_sample_size,i,loss.item()))

tm_name = "models/Cond/3features/TM-Final-UMNN_"+part+"_{}_{}_{}_{}_{:.2f}.pt".format(num_features,
          num_layers,num_hidden_features,training_sample_size,losses[-1])
torch.save(flow.state_dict(), tm_name)
print("trained model saved to {}".format(tm_name))
