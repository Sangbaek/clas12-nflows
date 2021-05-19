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

from utils.utilities import meter
from utils import make_histos
from utils.utilities import cartesian_converter
from utils.utilities import make_model
from utils import dataXZ

sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')
from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation



dev = "cuda:0" if torch.cuda.is_available() else "cpu"
#dev = "cpu"
print(dev)
device = torch.device(dev)

#reonstruct an nflow model
#model_path = "models/"
model_path = "models/Cond/16features/"
#model_name = "TM_16_18_4_400_299_-12.37.pt" #16 feature with Cond
#model_name = "TM_16_6_80_400_799_-26.48.pt" #16 feature with Cond
#model_name = "TMUMNN_16_6_80_400_499_-28.70.pt"
model_name = "TM-UMNN_16_6_80_400_3999_-42.91.pt"
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


params = model_name.split("_")
num_features = int(params[1])
num_layers = int(params[2])
num_hidden_features = int(params[3])
training_sample_size = int(params[4])

if "Final" in model_name:
    epoch_num = 4444 #identify as final
    training_loss = float((params[5]).split(".p")[0])
else:
    epoch_num = int(params[5])
    training_loss = float((params[6]).split(".p")[0])


print(num_features,num_layers,num_hidden_features,training_sample_size,training_loss)

flow, optimizer = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow.parameters()))
flow.load_state_dict(torch.load(model_path+model_name))
flow.eval()

maxloops = 500 #Number of overall loops
max_range = 10#Number of sets per loop
sample_size = 200 #Number of samples per set


#Initialize dataXZ object for quantile inverse transform
xz = dataXZ.dataXZ(feature_subset=feature_subset)
#QuantTran = xz.qt



for loop_num in range(maxloops):
    try:
        zs = []
        start = datetime.now()
        start_time = start.strftime("%H:%M:%S")
        print("Start Time =", start_time)
        for i in range(1,max_range+1):
            print("On set {}".format(i))
            
            #For nonconditional flows:
            #val_gen= flow.double().sample(sample_size).cpu().detach().numpy()
            
            #For conditional flows:
            sampleDict = xz.sample(sample_size)
            z = sampleDict["z"]
            z = z.detach().numpy()
            context_val = torch.tensor(z, dtype=torch.float32).to(device)
            val_gen = flow.sample(1,context=context_val).cpu().detach().numpy().reshape((sample_size,-1))


            zs.append(val_gen)
            now = datetime.now()
            elapsedTime = (now - start )
            print("Current time is {}".format(now.strftime("%H:%M:%S")))
            print("Elapsed time is {}".format(elapsedTime))
            print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(max_range+1-i)))

        X = np.concatenate(zs)
        #z = QuantTran.inverse_transform(X)

        df = pd.DataFrame(X)
        df.to_pickle("gendata/Cond/16features/UMNN/GenData_UMNN_{}_{}_{}_{}_{}_set_2_{}.pkl".format(num_features,
                num_layers,num_hidden_features,training_sample_size,training_loss,loop_num))
    except Exception as e:
        print("sorry, that didn't work, exception was:")
        print(e)


x_data = df[1]
y_data = df[2]
var_names = ["E Px","E Py"]
saveplots = False
output_dir = "."
title = "Px vs Py"
filename = title
units = ["GeV","Gev"]
ranges = [[-2,2,200],[-2,2,200]]

from matplotlib import interactive
make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                            saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                            filename=filename,units=units)
