#!/usr/bin/env python3
"""
A script to run nflow in HPC, like eofe cluster
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('pdf')
import sklearn.datasets as datasets
import itertools
import numpy as np

from datetime import datetime
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial import distance

import torch
from torch import nn
from torch import optim


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, QuantileTransformer
import pandas as pd

from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from utils import dataXZ

# Define device to be used
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

#read the data, with the defined data class
xz = dataXZ.dataXZ()
sampleDict = xz.sample(100000) #Get a subset of the datapoints
x = sampleDict["x"]
x = x.detach().numpy()

# #visualize the data
bin_size = [80,80]
fig, ax = plt.subplots(figsize =(10, 7)) 
plt.rcParams["font.size"] = "16"
ax.set_xlabel("Electron Momentum")  
ax.set_ylabel("Proton Momentum")
plt.title('Microphysics Simulated EP Distribution')

plt.hist2d(x[:,0], x[:,1],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
plt.xlim([1,6.5])
plt.ylim([0.2,1.1])
plt.colorbar()
plt.savefig("slurm/figures/raw_distribution_01.pdf")

fig, ax = plt.subplots(figsize =(10, 7)) 
plt.rcParams["font.size"] = "16"
ax.set_xlabel("Photon 1 Momentum")  
ax.set_ylabel("Photon 2 Momentum")
plt.title('Microphysics Simulated GG Distribution')
plt.hist2d(x[:,2], x[:,3],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
plt.xlim([1,9])
plt.ylim([0,5])
plt.colorbar()
plt.savefig("slurm/figures/raw_distribution_23.pdf")







#construct the model
num_features = 16

in_columns = num_features
out_columns = num_features

context_encoder = nn.Sequential(
          nn.Linear(num_features, 2*num_features),
          nn.ReLU(),
          nn.Linear(2*num_features, 2*num_features),
          nn.ReLU(),
          nn.Linear(2*num_features, 2*num_features)
        )

num_layers =6#12
#base_dist = StandardNormal(shape=[num_features]
base_dist = ConditionalDiagonalNormal(shape=[num_features],context_encoder=context_encoder)
#base_dist = DiagonalNormal(shape=[3])
transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=num_features))
    # transforms.append(MaskedAffineAutoregressiveTransform(features=num_features, 
    #                                                      hidden_features=100))
    transforms.append(MaskedAffineAutoregressiveTransform(features=num_features, 
                                                         hidden_features=80,
                                                          context_features=num_features))
    

    #transforms.append(MaskedUMNNAutoregressiveTransform(features=num_features, 
    #                                                      hidden_features=4))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist).to(device)
optimizer = optim.Adam(flow.parameters())
print("number of params: ", sum(p.numel() for p in flow.parameters()))













def plot_histo_1D(real_vals, gen_vals, label_real="Physics Data", label_gen="NFlow Model", col2 = "blue",title="Physics vs NFlow Models", saveloc=None):
    fig, axes = plt.subplots(1, 4, figsize=(4*5, 5))
    for INDEX, ax in zip((0, 1, 2,3 ), axes):
        _, bins, _ = ax.hist(real_vals[:, INDEX], bins=100, color = "red", label=label_real, density=True)
        ax.hist(gen_vals[:, INDEX], bins=bins, label=label_gen, color = col2,alpha=0.5, density=True)
        ax.legend(loc="lower left")
        ax.set_title("Feature {}".format(INDEX) )
    plt.tight_layout()
    if saveloc is not None: plt.savefig(saveloc)
    # plt.show()

def meter(dist1,dist2,feature):
  kld = entropy(dist1[:,feature],dist2[:,feature])
  emd = wasserstein_distance(dist1[:,feature],dist2[:,feature])
  jsd = distance.jensenshannon(dist1[:,feature],dist2[:,feature]) ** 2
  return [kld, emd, jsd]

num_iter = 1000
start_now = datetime.now()
start_time = start_now.strftime("%H:%M:%S")
print("Start Time =", start_time)
losses = []
f1_kd = []
f1_em = []
f1_js = []
f2_em = []
f3_em = []

for i in range(num_iter):
    sampleDict = xz.sample(100)
    x_train = sampleDict["x"][:, 0:num_features].to(device)
    z_train = sampleDict["z"][:, 0:num_features].to(device)

    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x_train,context=z_train).mean()
    #loss = -flow.log_prob(inputs=xxx,context=xxx).mean()
    #print(loss)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

    if i %100 == 0:
        run_time = datetime.now()
        elapsedTime = (run_time - start_now )
        print("On step {} - loss {:.2f}, Current Running Time = {:.2f} seconds".format(i,loss.item(),elapsedTime.total_seconds())) 

    if i == 5000 == 0:
            run_time = datetime.now()
            elapsedTime = (run_time - start_now )
            
            bbb = 50000
            z= flow.sample(bbb).cpu().detach().numpy()
            sampleDict = xz.sample(bbb)
            x = sampleDict["x"][:, 0:num_features] 
            x = x.detach().numpy()

            #plot_histo_1D(x,z)

            f1 = meter(x,z,0)
            f2 = meter(x,z,1)
            #f3 = meter(x,z,2)
            #f4 = meter(x,z,3)

            bin_size = [100,100]
            fig, ax = plt.subplots(figsize =(10, 7)) 
            plt.rcParams["font.size"] = "16"
            ax.set_xlabel("Electron Momentum")  
            ax.set_ylabel("Proton Momentum")
            plt.title('NFlow Generated EP Distribution')

            plt.hist2d(z[:,0], z[:,1],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
            #plt.xlim([-2,2])
            #plt.ylim([-2,2])
            plt.colorbar()
            plt.show()


            #if f1[1]*f2[1]*f3[1]*f4[1] < 1:
            print("On step {} - loss {:.2f}, Current Running Time = {:.2f} seconds".format(i,loss.item(),elapsedTime.total_seconds())) 
            #print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} F3: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),(f4[1]),))
            print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} F3: {:.5f} ".format((f1[1]),(f2[1]),(f2[1]),(f2[1]),))
            #if f1[1]*f2[1] < .001:
              #break

            f1_kd.append(f1[0])
            f1_em.append(f1[1])
            f1_js.append(f1[2])
            f2_em.append(f2[1])
            #f3_em.append(f3[1])
            #f4_em.append(f4[1])


now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print("End Time =", end_time)
elapsedTime = (now - start_now )
print("Total Run Time = {:.5f} seconds".format(elapsedTime.total_seconds()))


now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print("End Time =", end_time)
elapsedTime = (now - start_now )
print("Total Run Time = {:.5f} seconds".format(elapsedTime.total_seconds()))
    # if (i + 1) % 50 == 0:
    #     xline = torch.linspace(-1.5, 2.5)
    #     yline = torch.linspace(-.75, 1.25)
    #     xgrid, ygrid = torch.meshgrid(xline, yline)
    #     xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    #     with torch.no_grad():
    #         zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

    #     plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
    #     plt.title('iteration {}'.format(i + 1))
    #     plt.show()

#f1_kd = []
#f1_em = []
#f1_js = []

fig, ax = plt.subplots(figsize =(10, 7)) 
#print(np.arange(len(losses)))
plt.rcParams["font.size"] = "16"

plt.plot(np.arange(len(f1_em)),f1_em, '-b',label="Feature 0")
plt.plot(np.arange(len(f1_em)),f2_em, '-g',label="Feature 1")
plt.plot(np.arange(len(f1_em)),f3_em, '-r',label="Feature 2")
#plt.ylim([1000000000,0.0001])
ax.set_yscale('log')
plt.title('Wasserstein-1 Distance vs. Training Step')
ax.legend()
ax.set_xlabel("Training Step")  
ax.set_ylabel("Earth-Mover Distance")
plt.savefig("slurm/figures/EMD_training.pdf")


fig, ax = plt.subplots(figsize =(10, 7)) 
#print(np.arange(len(losses)))
plt.rcParams["font.size"] = "16"

plt.scatter(np.arange(len(f1_em)),f3_em, c='b', s=20)
#plt.ylim([1000000000,0.0001])
ax.set_yscale('log')
plt.title('Loss vs. Training Step')
ax.set_xlabel("Training Step")  
ax.set_ylabel("Loss")

fig, ax = plt.subplots(figsize =(10, 7)) 
#print(np.arange(len(losses)))
plt.rcParams["font.size"] = "16"

plt.scatter(np.arange(len(f1_js)),f1_js, c='g', s=20)
#plt.ylim([1000000000,0.0001])
#ax.set_yscale('log')
plt.title('Jensen–Shannon Divergence vs. Training Step')
ax.set_xlabel("Training Step")  
ax.set_ylabel("Jensen–Shannon Divergence")
plt.savefig("slurm/figures/JSD_training.pdf")

fig, ax = plt.subplots(figsize =(10, 7)) 
#print(np.arange(len(losses)))
plt.rcParams["font.size"] = "16"

plt.scatter(np.arange(len(f1_kd)),f1_kd, c='g', s=20)
#plt.ylim([1000000000,0.0001])
#ax.set_yscale('log')
plt.title('Kullback–Leibler Divergence vs. Training Step')
ax.set_xlabel("Training Step")  
ax.set_ylabel("Kullback–Leibler Divergence")
plt.savefig("slurm/figures/KLD_training.pdf")

#Testing

nsamp = 10000
sampleDict = xz.sample(nsamp) #Get a subset of the datapoints
x = sampleDict["x"]
x = x.detach().numpy()
z = sampleDict["z"]
z = z.detach().numpy()

context_val = torch.tensor(z, dtype=torch.float32).to(device)

val_gen = flow.sample(1,context=context_val).cpu().detach().numpy().reshape((nsamp,-1))


plt.hist(x[:,1],color = "red", density=True,bins=100)
plt.hist(val_gen[:,1],color = "black",alpha=0.5, density=True,bins=100)
plt.xlim([-3,3])


fig, ax = plt.subplots(figsize =(5, 3)) 
plt.hist2d(x[:,1], x[:,2],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
plt.colorbar()
plt.savefig("slurm/figures/validation_test_distribution.pdf")


fig, ax = plt.subplots(figsize =(5, 3)) 
plt.hist2d(val_gen[:,1], val_gen[:,2],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
plt.colorbar()
plt.savefig("slurm/figures/validation_nf_distribution.pdf")


