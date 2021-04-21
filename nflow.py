#!/usr/bin/env python3
"""
A script to run nflow in HPC, like eofe cluster
"""
import pickle5 as pickle
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
  def __init__(self, standard = False):
    with open('data/pi0.pkl', 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float32)
        #xz = xz[:, 1:]
        # z = xz[:, 16:]
        x = cartesian_converter(xz)
        # x = x[:, [0,4,8,12]]
        x = x[:, [3,7,11,15]]
        #x = xz[:, :16]
        #xwithoutPid = x[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]
        #xwithoutPid = x[:, [0,  4, 8, 12, ]]
        xwithoutPid = x
        #xwithoutPid = x[:, [0, 1, 4, 5, 8, 12, ]]
        # zwithoutPid = z[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]
        self.xz = xz
        self.x = torch.from_numpy(np.array(x))
        # self.z = torch.from_numpy(np.array(z))
        self.xwithoutPid = torch.from_numpy(xwithoutPid)
        # self.zwithoutPid = torch.from_numpy(zwithoutPid)

    if standard:
      self.standardize()

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
        randint = np.random.randint( self.xz.shape[0], size =n)
        xz = self.xz[randint]
        x = self.x[randint]
        # z = self.z[randint]
        xwithoutPid = self.xwithoutPid[randint]
        # zwithoutPid = self.zwithoutPid[randint]
        # return {"xz":xz, "x": x, "z": z, "xwithoutPid": xwithoutPid, "zwithoutPid": zwithoutPid}
        return {"xz":xz, "x": x, "xwithoutPid": xwithoutPid}

#returns an nx16 array, of energy, px, py, pz, for electron, proton, g1, g2
#You should just pass it the xz object from the dataXZ() class
def cartesian_converter(xznp):
  #split into electron, proton, gammas
  e_vec = xznp[:,1:5]
  p_vec = xznp[:,5:9]
  g1_vec = xznp[:,9:13]
  g2_vec = xznp[:,13:17]

  mass_e = .000511
  mass_p = 0.938
  mass_g = 0

  particles = [e_vec,p_vec,g1_vec,g2_vec]
  masses = [mass_e,mass_p,mass_g,mass_g]

  parts_new = []
  #convert from spherical to cartesian
  for part_vec, mass in zip(particles,masses):
    mom = part_vec[:,0]
    thet = part_vec[:,1]*np.pi/180
    phi = part_vec[:,2]*np.pi/180

    pz = mom*np.cos(thet)
    px = mom*np.sin(thet)*np.cos(phi)
    py = mom*np.sin(thet)*np.sin(phi)
    p2 = pz*pz+px*px+py*py
    E = np.sqrt(mass**2+p2)
    
    x_new = np.array([E,px,py,pz])
    parts_new.append(x_new)

  #reshape output into 1x16 arrays for each event
  e = parts_new[0]
  p = parts_new[1]
  g1 = parts_new[2]
  g2 = parts_new[3]
  out = np.concatenate((e.T,p.T,g1.T,g2.T), axis=1)

  return out

# Define device to be used
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

#read the data, with the defined data class
xz = dataXZ()
sampleDict = xz.sample(100000) #Get a subset of the datapoints
x = sampleDict["x"]
x = x.detach().numpy()

# #visualize the data
# bin_size = [80,80]
# fig, ax = plt.subplots(figsize =(10, 7)) 
# plt.rcParams["font.size"] = "16"
# ax.set_xlabel("Electron Momentum")  
# ax.set_ylabel("Proton Momentum")
# plt.title('Microphysics Simulated EP Distribution')

# plt.hist2d(x[:,0], x[:,1],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
# plt.xlim([1,6.5])
# plt.ylim([0.2,1.1])
# plt.colorbar()
# plt.savefig("raw_distribution_01.pdf")

# fig, ax = plt.subplots(figsize =(10, 7)) 
# plt.rcParams["font.size"] = "16"
# ax.set_xlabel("Photon 1 Momentum")  
# ax.set_ylabel("Photon 2 Momentum")
# plt.title('Microphysics Simulated GG Distribution')
# plt.hist2d(x[:,2], x[:,3],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
# plt.xlim([1,9])
# plt.ylim([0,5])
# plt.colorbar()
# plt.savefig("raw_distribution_23.pdf")

#construct the model
num_layers = 10#12
base_dist = StandardNormal(shape=[4])
#base_dist = DiagonalNormal(shape=[3])
transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=4))
    transforms.append(MaskedAffineAutoregressiveTransform(features=4, 
                                                          hidden_features=20))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist).to(device)
optimizer = optim.Adam(flow.parameters())
print("number of params: ", sum(p.numel() for p in flow.parameters()))

# def plot_histo_1D(real_vals, gen_vals, label_real="Physics Data", label_gen="NFlow Model", col2 = "blue",title="Physics vs NFlow Models", saveloc=None):
#     fig, axes = plt.subplots(1, 4, figsize=(4*5, 5))
#     for INDEX, ax in zip((0, 1, 2,3 ), axes):
#         _, bins, _ = ax.hist(real_vals[:, INDEX], bins=100, color = "red", label=label_real, density=True)
#         ax.hist(gen_vals[:, INDEX], bins=bins, label=label_gen, color = col2,alpha=0.5, density=True)
#         ax.legend(loc="lower left")
#         ax.set_title("Feature {}".format(INDEX) )
#     plt.tight_layout()
#     if saveloc is not None: plt.savefig(saveloc)
#     # plt.show()

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
    # x, y = datasets.make_moons(12, noise=.1)
    # x = torch.tensor(x, dtype=torch.float32)
    # print(x)
    # print(y)
    sampleDict = xz.sample(1000)
    x = sampleDict["x"][:, 0:4].to(device)
    #y = sampleDict["xwithoutPid"][:, 1:2] 
    #print(x)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())


    if i % 10 == 0:
        run_time = datetime.now()
        elapsedTime = (run_time - start_now )
        
        bbb = 10000
        z= flow.sample(bbb).cpu().detach().numpy()
        sampleDict = xz.sample(bbb)
        x = sampleDict["x"][:, 0:4]
        x = x.detach().numpy()

        #plot_histo_1D(x,z)

        f1 = meter(x,z,0)
        f2 = meter(x,z,1)
        f3 = meter(x,z,2)
        f4 = meter(x,z,3)


        if f1[1]*f2[1]*f3[1]*f4[1] < 1:
          print("On step {} - loss {:.2f}, Current Running Time = {:.2f} seconds".format(i,loss.item(),elapsedTime.total_seconds())) 
          print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} F3: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),(f4[1]),))
          #break

        f1_kd.append(f1[0])
        f1_em.append(f1[1])
        f1_js.append(f1[2])
        f2_em.append(f2[1])
        f3_em.append(f3[1])

        if i % 100 == 0:
          bbb = 100000
          zzz= flow.sample(bbb).cpu().detach().numpy()
          sampleDictzz = xz.sample(bbb)
          x = sampleDict["x"]
          x = x.detach().numpy()
          # plot_histo_1D(x,z, saveloc="training_step{}.pdf".format(i))
          # print("On step {} - loss {:.2f}, Current Running Time = {:.2f} seconds".format(i,loss.item(),elapsedTime.total_seconds())) 
          print("KL Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ".format((f1[0]),(f2[0]),(f3[0]),))
          # print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),))
          #print("JS Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ".format((f1[2]),(f2[2]),(f3[2]),))


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

# fig, ax = plt.subplots(figsize =(10, 7)) 
# #print(np.arange(len(losses)))
# plt.rcParams["font.size"] = "16"

# plt.plot(np.arange(len(f1_em)),f1_em, '-b',label="Feature 0")
# plt.plot(np.arange(len(f1_em)),f2_em, '-g',label="Feature 1")
# plt.plot(np.arange(len(f1_em)),f3_em, '-r',label="Feature 2")
# #plt.ylim([1000000000,0.0001])
# ax.set_yscale('log')
# plt.title('Wasserstein-1 Distance vs. Training Step')
# ax.legend()
# ax.set_xlabel("Training Step")  
# ax.set_ylabel("Earth-Mover Distance")
# plt.savefig("EMD_training.pdf")


# fig, ax = plt.subplots(figsize =(10, 7)) 
# #print(np.arange(len(losses)))
# plt.rcParams["font.size"] = "16"

# plt.scatter(np.arange(len(f1_em)),f3_em, c='b', s=20)
# #plt.ylim([1000000000,0.0001])
# ax.set_yscale('log')
# plt.title('Loss vs. Training Step')
# ax.set_xlabel("Training Step")  
# ax.set_ylabel("Loss")

# fig, ax = plt.subplots(figsize =(10, 7)) 
# #print(np.arange(len(losses)))
# plt.rcParams["font.size"] = "16"

# plt.scatter(np.arange(len(f1_js)),f1_js, c='g', s=20)
# #plt.ylim([1000000000,0.0001])
# #ax.set_yscale('log')
# plt.title('Jensen–Shannon Divergence vs. Training Step')
# ax.set_xlabel("Training Step")  
# ax.set_ylabel("Jensen–Shannon Divergence")
# plt.savefig("JSD_training.pdf")

# fig, ax = plt.subplots(figsize =(10, 7)) 
# #print(np.arange(len(losses)))
# plt.rcParams["font.size"] = "16"

# plt.scatter(np.arange(len(f1_kd)),f1_kd, c='g', s=20)
# #plt.ylim([1000000000,0.0001])
# #ax.set_yscale('log')
# plt.title('Kullback–Leibler Divergence vs. Training Step')
# ax.set_xlabel("Training Step")  
# ax.set_ylabel("Kullback–Leibler Divergence")
# plt.savefig("KLD_training.pdf")

#Testing

aa = flow.sample(100000).cpu().detach().numpy()
# plt.scatter(aa[:,0], aa[:,1], c='r', s=5, alpha=0.5)
# plt.savefig("test_sample.pdf")

z = aa

bbb = 100000
z= flow.sample(bbb).cpu().detach().numpy()
sampleDict = xz.sample(bbb)
sampleDict2 = xz.sample(bbb)
y = sampleDict2["x"]
y = y.detach().numpy()
x = sampleDict["x"]
x = x.detach().numpy()

# plot_histo_1D(x,z, saveloc="training_xz.pdf")
# plot_histo_1D(x,y,label_real="Physics Sample 1", label_gen="Physics Sample 2",col2="green", saveloc="training_xy.pdf")

f1 = meter(x,z,0)
f2 = meter(x,z,1)
f3 = meter(x,z,2)
f4 = meter(x,z,3)

print("Values for Physics Data vs. NFlow Model:")
print("KL Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f}  ".format((f1[0]),(f2[0]),(f3[0]),(f4[0])))
print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),(f4[1])))
print("JS Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f}  ,F3: {:.5f} ".format((f1[2]),(f2[2]),(f3[2]),(f4[2])))
print('\n')

f1 = [i / j for i, j in zip(f1,meter(x,y,0))]
f2 = [i / j for i, j in zip(f2,meter(x,y,1))]
f3 = [i / j for i, j in zip(f3,meter(x,y,2))]
f4 = [i / j for i, j in zip(f4,meter(x,y,3))]

print("Ratio of KL, EM, and JS values from NFlow comparision and two physics model samples:")
print("KL Divergence Ratio: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f}  ".format((f1[0]),(f2[0]),(f3[0]),(f4[0])))
print("EM Distance   Ratio: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),(f4[1])))
print("JS Divergence Ratio: F0: {:.5f}  F1: {:.5f}  F2: {:.5f}  ,F3: {:.5f} ".format((f1[2]),(f2[2]),(f3[2]),(f4[2])))
print('\n')

f1 = meter(x,y,0)
f2 = meter(x,y,1)
f3 = meter(x,y,2)
f4x = meter(x,y,3)

print("Values for two samples from physics data")
print("KL Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f}  ".format((f1[0]),(f2[0]),(f3[0]),(f4x[0])))
print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),(f4x[1])))
print("JS Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f}  ,F3: {:.5f} ".format((f1[2]),(f2[2]),(f3[2]),(f4x[2])))


