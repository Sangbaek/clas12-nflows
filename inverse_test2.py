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

from utils.utilities import meter
from utils import make_histos
from utils.utilities import cartesian_converter
from utils.utilities import make_model
from utils import dataXZ

from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(dev)
device = torch.device(dev)

#reonstruct an nflow model
#model_path = "models/"
model_path = "models/Cond/3features/"
feature_subset = [9,10,11,13,14,15] #All 16 features

print(" reading photon NF model ")
model_name = "TM-Final-UMNN_phot_3_10_80_128_-5.38.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

print(num_features,num_layers,num_hidden_features,training_sample_size)

flow_g, optimizer_g = make_model(num_layers,num_features,num_hidden_features,device)

print("number of params: ", sum(p.numel() for p in flow_g.parameters()))
flow_g.load_state_dict(torch.load(model_path+model_name))
flow_g.eval()

print(" reading photon2 NF model ")
model_name = "TM-Final-UMNN_phot2_3_10_80_128_-6.04.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

print(num_features,num_layers,num_hidden_features,training_sample_size)

flow_g2, optimizer_g2 = make_model(num_layers,num_features,num_hidden_features,device)

print("number of params: ", sum(p.numel() for p in flow_g2.parameters()))
flow_g2.load_state_dict(torch.load(model_path+model_name))
flow_g2.eval()


print("reading truth data")
train = dataXZ.dataXZ(feature_subset=feature_subset, file = "data/epgg_train.pkl", mode = "epgg")
truth_entire = train.truth.detach().numpy()
reco_entire = train.reco.detach().numpy()
print("done with reading truth data")

print("reading test data")
test = dataXZ.dataX(feature_subset=feature_subset, file = "data/epgg_test.pkl", mode = "epgg")
reco_test = test.reco.detach().numpy()
print("done with reading test data")

trials = 10000 #Number of overall loops

means = np.mean(reco_entire - truth_entire, axis = 1)
stds = np.std(reco_entire - truth_entire, axis = 1)

n_sample = 100
n_loop = len(reco_test)//n_sample

for loop_num in range(n_loop):
	print("new loop "+str(loop_num))

	truths_guess = []

	start = datetime.now()
	start_time = start.strftime("%H:%M:%S")
	print("Start Time =", start_time)

	for i in range(n_sample):

		maxtruth_g = []
		logprob_g = []
		maxtruth_g2 = []
		logprob_g2 = []

		#photon
		reco_g = reco_test[i+n_sample*loop_num:i+n_sample*loop_num+1, [0,1,2]]
		mean_g = means[[0,1,2]]
		std_g = stds[[0,1,2]]

		#photon2
		reco_g2 = reco_test[i+n_sample*loop_num:i+n_sample*loop_num+1, [3,4,5]]
		mean_g2 = means[[3,4,5]]
		std_g2 = stds[[3,4,5]]

		truth_g = reco_g + np.random.normal(mean_g, std_g, (trials, 3))
		truth_g2 = reco_g2 + np.random.normal(mean_g2, std_g2, (trials, 3))

		# for reco in reco_g:
		reco_useful = np.tile(reco_g, (trials, 1))
		reco_useful = torch.tensor(reco_useful, dtype=torch.float32).to(device)
		truth_g = torch.tensor(truth_g, dtype=torch.float32).to(device)
		logprob = flow_g.log_prob(inputs=reco_useful,context=truth_g)
		ind_max = np.argmax(logprob.cpu().detach().numpy())
		maxtruth_g.append(truth_g[ind_max:ind_max+1, :])
		logprob_g.append(logprob[ind_max])

		# for reco in reco_g:
		reco_useful = np.tile(reco_g2, (trials, 1))
		reco_useful = torch.tensor(reco_useful, dtype=torch.float32).to(device)
		truth_g2 = torch.tensor(truth_g2, dtype=torch.float32).to(device)
		logprob = flow_g2.log_prob(inputs=reco_useful,context=truth_g2)
		ind_max = np.argmax(logprob.cpu().detach().numpy())
		maxtruth_g2.append(truth_g[ind_max:ind_max+1, :])
		logprob_g2.append(logprob[ind_max])

		#photon
		truth_val_g = maxtruth_g[np.argmax(logprob_g)].cpu().detach().numpy()
		E_true_g = np.sqrt(truth_val_g[:, 0]**2 + truth_val_g[:, 1]**2  + truth_val_g[:, 2]**2).reshape((-1, 1))

		#photon
		truth_val_g2 = maxtruth_g2[np.argmax(logprob_g2)].cpu().detach().numpy()
		E_true_g2 = np.sqrt(truth_val_g2[:, 0]**2 + truth_val_g2[:, 1]**2  + truth_val_g2[:, 2]**2).reshape((-1, 1))

		NF_true = np.hstack( (E_true_g, truth_val_g, E_true_g2, truth_val_g2))
		truths_guess.append(NF_true)

	now = datetime.now()
	elapsedTime = (now - start )
	print("Current time is {}".format(now.strftime("%H:%M:%S")))
	print("Elapsed time is {}".format(elapsedTime))
	# print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(max_range+1-i)))
	Truths = np.concatenate(truths_guess)
	df_Truths = pd.DataFrame(Truths)
	df_Truths.to_pickle("gendata/Cond/3features/UMNN/Truths_Test_pi0_{}.pkl".format(loop_num))#num_features,
	#     num_layers,num_hidden_features,training_sample_size,loop_num))
	#print(truths_guess )
	#print(truth_validation[0:n_sample, :])
	#print(reco_test[0:n_sample, :])
print("done")
quit()
