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
feature_subset = "all" #All 16 features

print(" reading electron NF model ")
model_name = "TM-Final-UMNN_elec_3_6_80_128_-9.54.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

print(num_features,num_layers,num_hidden_features,training_sample_size,training_loss)

flow_e, optimizer_e = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow_e.parameters()))
flow_e.load_state_dict(torch.load(model_path+model_name))
flow_e.eval()

print(" reading proton NF model ")
model_name = "TM-Final-UMNN_prot_3_6_80_128_-9.97.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

print(num_features,num_layers,num_hidden_features,training_sample_size,training_loss)

flow_p, optimizer_p = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow_p.parameters()))
flow_p.load_state_dict(torch.load(model_path+model_name))
flow_p.eval()


print(" reading photon NF model ")
model_name = "TM-Final-UMNN_phot_3_2_80_128_-6.81.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

print(num_features,num_layers,num_hidden_features,training_sample_size,training_loss)

flow_g, optimizer_g = make_model(num_layers,num_features,num_hidden_features,device)

print("number of params: ", sum(p.numel() for p in flow_g.parameters()))
flow_g.load_state_dict(torch.load(model_path+model_name))
flow_g.eval()


print("reading truth data")
train = dataXZ.dataXZ(feature_subset=feature_subset, file = "data/train.pkl", mode = "epg")
truth_entire = train.truth.detach().numpy()
print("done with reading truth data")

print("reading validation data")
validation = dataXZ.dataXZ(feature_subset=feature_subset, file = "data/validation.pkl", mode = "epg")
reco_validation = validation.reco.detach().numpy()
print("done with reading validation data")

max_range = 10#Number of sets per loop
sample_size = 200 #Number of samples per set
maxloops = len(xentire)//(max_range*sample_size) #Number of overall loops

for loop_num in range(maxloops):
    print("new loop "+str(loop_num))
    truths_guess = []
    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    print("Start Time =", start_time)
    for i in range(1,max_range+1):
        print("On set {}".format(i))

        #electron
        truth_e = torch.tensor(truth_entire[:, [1,2,3]], dtype=torch.float32).to(device)
        #proton
        truth_p = torch.tensor(truth_entire[:, [5,6,7]], dtype=torch.float32).to(device)
        #photon
        truth_g = torch.tensor(truth_entire[:, [9,10,11]], dtype=torch.float32).to(device)

        #electron
        reco_e = reco_validation[sample_size*(max_range*loop_num+i-1):sample_size*(max_range*loop_num+i), [1,2,3]]
        #proton
        reco_p = reco_validation[sample_size*(max_range*loop_num+i-1):sample_size*(max_range*loop_num+i), [5,6,7]]
        #photon
        reco_g = reco_validation[sample_size*(max_range*loop_num+i-1):sample_size*(max_range*loop_num+i), [9,10,11]]


        for reco in reco_e:
            reco_useful = np.tile(reco, (len(truth_e), 1))
            reco_useful = torch.tensor(reco_useful, dtype=torch.float32).to(device)
            logprob = flow_e.log_prob(inputs=reco_useful,context=truth_e)
            ind_max = np.argmax(logprob)
            truth = truth_e[ind_max, [1, 2, 3]].cpu().detach().numpy()
            truth_val_e.append(truth)

        for reco in reco_p:
            reco_useful = np.tile(reco, (len(truth_p), 1))
            reco_useful = torch.tensor(reco_useful, dtype=torch.float32).to(device)
            logprob = flow_p.log_prob(inputs=reco_useful,context=truth_p)
            ind_max = np.argmax(logprob)
            truth = truth_p[ind_max, [5, 6, 7]].cpu().detach().numpy()
            truth_val_p.append(truth)

        for reco in reco_g:
            reco_useful = np.tile(reco, (len(truth_g), 1))
            reco_useful = torch.tensor(reco_useful, dtype=torch.float32).to(device)
            logprob = flow_g.log_prob(inputs=reco_useful,context=truth_g)
            ind_max = np.argmax(logprob)
            truth = truth_g[ind_max, [9, 10, 11]].cpu().detach().numpy()
            truth_val_g.append(truth)

        #electron
        truth_val_e = np.array(truth_val_e)
        E_true_e = np.sqrt(truth_val_e[:, 0]**2 + truth_val_e[:, 1]**2  + truth_val_e[:, 2]**2  +  (0.5109989461 * 0.001)**2).reshape((-1, 1))
        #proton
        truth_val_p = np.array(truth_val_p)
        E_true_p = np.sqrt(truth_val_p[:, 0]**2 + truth_val_p[:, 1]**2  + truth_val_p[:, 2]**2 + (0.938272081)**2).reshape((-1, 1))
        #photon
        truth_val_g = np.array(truth_val_g)
        E_true_g = np.sqrt(truth_val_g[:, 0]**2 + truth_val_g[:, 1]**2  + truth_val_g[:, 2]**2).reshape((-1, 1))

        NF_true = np.hstack( (E_gen_e, val_gen_e, E_gen_p, val_gen_p, E_gen_g, val_gen_g))
        truths_guess.append(NF_true)

        now = datetime.now()
        elapsedTime = (now - start )
        print("Current time is {}".format(now.strftime("%H:%M:%S")))
        print("Elapsed time is {}".format(elapsedTime))
        print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(max_range+1-i)))
    Truths = np.concatenate(truths_guess)
    df_Truths = pd.DataFrame(Truths)
    df_Truths.to_pickle("gendata/Cond/3features/UMNN/Truths_UMNN_{}_{}_{}_{}_{}_dvcs_{}.pkl".format(num_features,
            num_layers,num_hidden_features,training_sample_size,training_loss,loop_num))

print("done")
quit()