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
feature_subset = [1,2,3,5,6,7,9,10,11] #All 16 features

print(" reading electron NF model ")
model_name = "TM-Final-UMNN_elec_3_6_80_128_-9.54.pt"
model_name ="TM-Final-UMNN_elec_3_6_80_128_-8.26.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

print(num_features,num_layers,num_hidden_features,training_sample_size)

flow_e, optimizer_e = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow_e.parameters()))
flow_e.load_state_dict(torch.load(model_path+model_name))
flow_e.eval()

print(" reading proton NF model ")
model_name = "TM-Final-UMNN_prot_3_6_80_128_-9.97.pt"
model_name = "TM-Final-UMNN_prot_3_6_80_128_-9.23.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

print(num_features,num_layers,num_hidden_features,training_sample_size)

flow_p, optimizer_p = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow_p.parameters()))
flow_p.load_state_dict(torch.load(model_path+model_name))
flow_p.eval()


print(" reading photon NF model ")
model_name = "TM-Final-UMNN_phot_3_2_80_128_-6.81.pt"
model_name = "TM-Final-UMNN_phot_3_6_80_128_-6.43.pt"
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


print("reading truth data")
train = dataXZ.dataXZ(feature_subset=feature_subset, file = "data/train.pkl", mode = "epg")
truth_entire = train.truth.detach().numpy()
reco_entire = train.reco.detach().numpy()
print("done with reading truth data")

print("reading validation data")
validation = dataXZ.dataXZ(feature_subset=feature_subset, file = "data/validation.pkl", mode = "epg")
truth_validation = validation.truth.detach().numpy()
reco_validation = validation.reco.detach().numpy()
print("done with reading validation data")

trials = 10000 #Number of overall loops

means = np.mean(reco_entire - truth_entire, axis = 1)
stds = np.std(reco_entire - truth_entire, axis = 1)

n_sample = 100
n_loop = len(reco_validation)//n_sample

for loop_num in range(n_loop):
	print("new loop "+str(loop_num))

	truths_guess = []

	start = datetime.now()
	start_time = start.strftime("%H:%M:%S")
	print("Start Time =", start_time)

	for i in range(n_sample):

		maxtruth_e = []
		logprob_e = []
		maxtruth_p = []
		logprob_p = []
		maxtruth_g = []
		logprob_g = []

		#electron
		reco_e = reco_validation[i+n_sample*loop_num:i+n_sample*loop_num+1, [0,1,2]]
		mean_e = means[[0,1,2]]
		std_e = stds[[0,1,2]]
		#proton
		reco_p = reco_validation[i+n_sample*loop_num:i+n_sample*loop_num+1, [3,4,5]]
		mean_p = means[[3,4,5]]
		std_p = stds[[3,4,5]]
		#photon
		reco_g = reco_validation[i+n_sample*loop_num:i+n_sample*loop_num+1, [6,7,8]]
		mean_g = means[[6,7,8]]
		std_g = stds[[6,7,8]]

		truth_e = reco_e + np.random.normal(mean_e, std_e, (trials, 3))
		truth_p = reco_p + np.random.normal(mean_p, std_p, (trials, 3))
		truth_g = reco_g + np.random.normal(mean_g, std_g, (trials, 3))

		# for reco in reco_e:
		reco_useful = np.tile(reco_e, (trials, 1))
		reco_useful = torch.tensor(reco_useful, dtype=torch.float32).to(device)
		truth_e = torch.tensor(truth_e, dtype=torch.float32).to(device)
		logprob = flow_e.log_prob(inputs=reco_useful,context=truth_e)
		ind_max = np.argmax(logprob.cpu().detach().numpy())
		maxtruth_e.append(truth_e[ind_max:ind_max+1, :])
		logprob_e.append(logprob[ind_max])

		# for reco in reco_p:
		reco_useful = np.tile(reco_p, (trials, 1))
		reco_useful = torch.tensor(reco_useful, dtype=torch.float32).to(device)
		truth_p = torch.tensor(truth_p, dtype=torch.float32).to(device)
		logprob = flow_p.log_prob(inputs=reco_useful,context=truth_p)
		ind_max = np.argmax(logprob.cpu().detach().numpy())
		maxtruth_p.append(truth_p[ind_max:ind_max+1, :])
		logprob_p.append(logprob[ind_max])

		# for reco in reco_g:
		reco_useful = np.tile(reco_g, (trials, 1))
		reco_useful = torch.tensor(reco_useful, dtype=torch.float32).to(device)
		truth_g = torch.tensor(truth_g, dtype=torch.float32).to(device)
		logprob = flow_g.log_prob(inputs=reco_useful,context=truth_g)
		ind_max = np.argmax(logprob.cpu().detach().numpy())
		maxtruth_g.append(truth_g[ind_max:ind_max+1, :])
		logprob_g.append(logprob[ind_max])

		#electron
		truth_val_e = maxtruth_e[np.argmax(logprob_e)].cpu().detach().numpy()
		E_true_e = np.sqrt(truth_val_e[:, 0]**2 + truth_val_e[:, 1]**2  + truth_val_e[:, 2]**2  +  (0.5109989461 * 0.001)**2).reshape((-1, 1))
		#proton
		truth_val_p = maxtruth_p[np.argmax(logprob_p)].cpu().detach().numpy()
		E_true_p = np.sqrt(truth_val_p[:, 0]**2 + truth_val_p[:, 1]**2  + truth_val_p[:, 2]**2 + (0.938272081)**2).reshape((-1, 1))
		#photon
		truth_val_g = maxtruth_g[np.argmax(logprob_g)].cpu().detach().numpy()
		E_true_g = np.sqrt(truth_val_g[:, 0]**2 + truth_val_g[:, 1]**2  + truth_val_g[:, 2]**2).reshape((-1, 1))

		NF_true = np.hstack( (E_true_e, truth_val_e, E_true_p, truth_val_p, E_true_g, truth_val_g))
		truths_guess.append(NF_true)

	now = datetime.now()
	elapsedTime = (now - start )
	print("Current time is {}".format(now.strftime("%H:%M:%S")))
	print("Elapsed time is {}".format(elapsedTime))
	# print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(max_range+1-i)))
	Truths = np.concatenate(truths_guess)
	df_Truths = pd.DataFrame(Truths)
	df_Truths.to_pickle("gendata/Cond/3features/UMNN/Truths_dvcs_{}.pkl".format(loop_num))#num_features,
	#     num_layers,num_hidden_features,training_sample_size,loop_num))
	#print(truths_guess )
	#print(truth_validation[0:n_sample, :])
	#print(reco_validation[0:n_sample, :])
print("done")
quit()
