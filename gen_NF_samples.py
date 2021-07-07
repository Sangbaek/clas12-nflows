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
#dev = "cpu"
print(dev)
device = torch.device(dev)

#reonstruct an nflow model
#model_path = "models/"
model_path = "models/Cond/3features/"
#model_name = "TM_16_18_4_400_299_-12.37.pt" #16 feature with Cond
#model_name = "TM_16_6_80_400_799_-26.48.pt" #16 feature with Cond
#model_name = "TMUMNN_16_6_80_400_499_-28.70.pt"
#model_name = "TM-UMNN_16_6_80_400_3999_-42.91.pt"
feature_subset = "all" #All 16 features



#model_name = "TM_16_18_20_100_799_-15.19.pt" #For initial double precision studies
#model_name = "TM_4_6_4_100_3199_-0.88.pt" #4 features with QD, initial training

#model_name = "TM_16_16_32_400_4399_-14.42.pt" #16 feature with QD
#feature_subset = "all" #All 16 features

#model_name = "TM-Final_4_6_80_400_-1.97.pt" #4 feature (electron) train, done 5/10 at 4 PM
#feature_subset = [0,1,2,3] #Just electron features


#This mechanism needs to be adjusted. It is hard coded. 
# A mechanism to read the feature subset from the trained model should be implemented
#feature_subset = [4,5,6,7] #Just proton features

print(" reading electron NF model ")
#model_name = "TM-Final-UMNN_elec_4_6_80_400_-9.76.pt"
model_name = "TM-Final-UMNN_elec_3_6_80_400_-8.08.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

if "Final" in model_name:
    epoch_num = 9999 #identify as final
    training_loss = float((params[6]).split(".p")[0])
else:
    epoch_num = int(params[5])
    training_loss = float((params[6]).split(".p")[0])


print(num_features,num_layers,num_hidden_features,training_sample_size,training_loss)

flow_e, optimizer_e = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow_e.parameters()))
flow_e.load_state_dict(torch.load(model_path+model_name))
flow_e.eval()

print(" reading proton NF model ")

#model_name = "TM-Final-UMNN_prot_4_6_80_400_-13.90.pt"
model_name = "TM-Final-UMNN_prot_3_6_40_400_-7.74.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

if "Final" in model_name:
    epoch_num = 9999 #identify as final
    training_loss = float((params[6]).split(".p")[0])
else:
    epoch_num = int(params[5])
    training_loss = float((params[6]).split(".p")[0])


print(num_features,num_layers,num_hidden_features,training_sample_size,training_loss)

flow_p, optimizer_p = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow_p.parameters()))
flow_p.load_state_dict(torch.load(model_path+model_name))
flow_p.eval()


print(" reading photon NF model ")


#model_name = "TM-Final-UMNN_phot_4_6_40_400_-10.24.pt"
model_name = "TM-Final-UMNN_phot_3_6_40_400_-3.96.pt"
params = model_name.split("_")
num_features = int(params[2])
num_layers = int(params[3])
num_hidden_features = int(params[4])
training_sample_size = int(params[5])

if "Final" in model_name:
    epoch_num = 9999 #identify as final
    training_loss = float((params[6]).split(".p")[0])
else:
    epoch_num = int(params[5])
    training_loss = float((params[6]).split(".p")[0])


print(num_features,num_layers,num_hidden_features,training_sample_size,training_loss)

flow_g, optimizer_g = make_model(num_layers,num_features,num_hidden_features,device)

print("number of params: ", sum(p.numel() for p in flow_g.parameters()))
flow_g.load_state_dict(torch.load(model_path+model_name))
flow_g.eval()


print (" reading data")
#Initialize dataXZ object for quantile inverse transform
xz = dataXZ.dataXZ(feature_subset=feature_subset, file = "data/pi0toepg.pkl", mode = "epg")
xentire = xz.x.detach().numpy()
zentire = xz.z.detach().numpy()
#QuantTran = xz.qt

print (" done with reading data")

max_range = 10#Number of sets per loop
sample_size = 200 #Number of samples per set
maxloops = len(xentire)//(max_range*sample_size) #Number of overall loops

for loop_num in range(maxloops):
    print("new loop "+str(loop_num))
    xs = []
    zs = []
    gens = []
    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    print("Start Time =", start_time)
    for i in range(1,max_range+1):
        print("On set {}".format(i))
        
        #For nonconditional flows:
        #val_gen= flow.double().sample(sample_size).cpu().detach().numpy()
        
        #For conditional flows:
        #z = sampleDict["z"]
        #x = sampleDict["x"]
        z = zentire[sample_size*(max_range*loop_num+i-1):sample_size*(max_range*loop_num+i), :]
        x = xentire[sample_size*(max_range*loop_num+i-1):sample_size*(max_range*loop_num+i), :]
        zs.append(z)
        xs.append(x)
        #electron
        context_val_e = torch.tensor(z[:, [1,2,3]], dtype=torch.float32).to(device)
        val_gen_e = flow_e.sample(1,context=context_val_e).cpu().detach().numpy().reshape((sample_size,-1))
        E_gen_e = np.sqrt(val_gen_e[:, 0]**2 + (0.5109989461 * 0.001)**2).reshape((-1, 1))
        #proton
        context_val_p = torch.tensor(z[:, [5,6,7]], dtype=torch.float32).to(device)
        val_gen_p = flow_p.sample(1,context=context_val_p).cpu().detach().numpy().reshape((sample_size,-1))
        E_gen_p = np.sqrt(val_gen_p[:, 0]**2 + (0.938272081)**2).reshape((-1, 1))
        #photon
        context_val_g = torch.tensor(z[:, [9,10,11]], dtype=torch.float32).to(device)
        val_gen_g = flow_g.sample(1,context=context_val_g).cpu().detach().numpy().reshape((sample_size,-1))
        E_gen_g = val_gen_g[:, 0].reshape((-1, 1))
        gens.append( np.hstack( (E_gen_e, val_gen_e, E_gen_p, val_gen_p, E_gen_g, val_gen_g)))
        now = datetime.now()
        elapsedTime = (now - start )
        print("Current time is {}".format(now.strftime("%H:%M:%S")))
        print("Elapsed time is {}".format(elapsedTime))
        print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(max_range+1-i)))
    X = np.concatenate(xs)
    Z = np.concatenate(zs)
    Gens = np.concatenate(gens)
    #z = QuantTran.inverse_transform(X)
    df_Z = pd.DataFrame(Z)
    df_Z.to_pickle("gendata/Cond/3features/UMNN/Z_UMNN_{}_{}_{}_{}_{}_pi0_{}.pkl".format(num_features,
            num_layers,num_hidden_features,training_sample_size,training_loss,loop_num))
    df_X = pd.DataFrame(X)
    df_X.to_pickle("gendata/Cond/3features/UMNN/X_UMNN_{}_{}_{}_{}_{}_pi0_{}.pkl".format(num_features,
            num_layers,num_hidden_features,training_sample_size,training_loss,loop_num))
    df_Gens = pd.DataFrame(Gens)
    df_Gens.to_pickle("gendata/Cond/3features/UMNN/GenData_UMNN_{}_{}_{}_{}_{}_pi0_{}.pkl".format(num_features,
            num_layers,num_hidden_features,training_sample_size,training_loss,loop_num))

print("done")
quit()
