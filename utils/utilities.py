import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import sys

import torch
from torch import nn
from torch import optim

sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')
from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from sklearn.model_selection import train_test_split

def split_data(data,test_size=0.2,random_state=42):
    data_train, data_test = train_test_split(data, test_size=test_size, random_state=random_state) 
    return data_train, data_test

def make_model(num_layers,num_features,num_hidden_features,device):


    context_encoder = nn.Sequential(
            nn.Linear(num_features, 2*num_features),
            nn.ReLU(),
            nn.Linear(2*num_features, 2*num_features),
            nn.ReLU(),
            nn.Linear(2*num_features, 2*num_features)
            )

    base_dist = ConditionalDiagonalNormal(shape=[num_features],context_encoder=context_encoder)

    #base_dist = StandardNormal(shape=[num_features])
    #base_dist = DiagonalNormal(shape=[num_features])
    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=num_features))
        #UMNN
        #transforms.append(MaskedUMNNAutoregressiveTransform(features=num_features, 
        #                                                    hidden_features=num_hidden_features)) 

        #Conditional MAA
        #transforms.append(MaskedAffineAutoregressiveTransform(features=num_features, 
        #                                                 hidden_features=num_hidden_features,
        #                                                  context_features=num_features))
        
        #Conditional UMNN
        transforms.append(MaskedUMNNAutoregressiveTransform(features=num_features, 
                                                              hidden_features=num_hidden_features,
                                                              context_features=num_features))
    transform = CompositeTransform(transforms)
    
    #Uncomment the below if float64
    #flow = Flow(transform, base_dist).double().to(device)
    #Uncomment the below if float32
    flow = Flow(transform, base_dist).to(device)
    
    optimizer = optim.Adam(flow.parameters())

    return (flow,optimizer)

def meter(dist1,dist2,feature):
  kld = entropy(dist1[:,feature],dist2[:,feature])
  emd = wasserstein_distance(dist1[:,feature],dist2[:,feature])
  jsd = distance.jensenshannon(dist1[:,feature],dist2[:,feature]) ** 2
  return [emd, kld, jsd]

#returns an nx16 array, of energy, px, py, pz, for electron, proton, g1, g2
#You should just pass it the xz object from the dataXZ() class
def cartesian_converter(xznp,type='x'):
  #split into electron, proton, gammas
  if type=='x':
    e_vec = xznp[:,1:5]
    p_vec = xznp[:,5:9]
    g1_vec = xznp[:,9:13]
    g2_vec = xznp[:,13:17]
  if type=='z':
    e_vec = xznp[:,17:21]
    p_vec = xznp[:,21:25]
    g1_vec = xznp[:,25:29]
    g2_vec = xznp[:,29:33]


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


def dot(vec1, vec2):
    # dot product of two 3d vectors
    return vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]

def mag(vec1):
    # L2 norm of vector
    return np.sqrt(dot(vec1, vec1))

def mag2(vec1):
    # square of L2 norm
    return  dot(vec1, vec1)

def cosTheta(vec1, vec2):
    # cosine angle between two 3d vectors
    return dot(vec1,vec2)/np.sqrt(mag2(vec1) * mag2(vec2))

def angle(vec1, vec2):
    # angle between two 3d vectors
    return 180/np.pi*np.arccos(np.minimum(1, cosTheta(vec1, vec2)))

def cross(vec1, vec2):
    # cross product of two 3d vectors
    return [vec1[1]*vec2[2]-vec1[2]*vec2[1], vec1[2]*vec2[0]-vec1[0]*vec2[2], vec1[0]*vec2[1]-vec1[1]*vec2[0]]

def vecAdd(gam1, gam2):
    # add two 3d vectors
    return [gam1[0]+gam2[0], gam1[1]+gam2[1], gam1[2]+gam2[2]]

def pi0Energy(gam1, gam2):
    # reconstructed pi0 energy of two 3d photon momenta
    return mag(gam1)+mag(gam2)

def pi0InvMass(gam1, gam2):
    # pi0 invariant mass of two 3d photon momenta
    pi0mass2 = pi0Energy(gam1, gam2)**2-mag2(vecAdd(gam1, gam2))
    pi0mass2 = np.where(pi0mass2 >= 0, pi0mass2, 10**6)
    pi0mass = np.sqrt(pi0mass2)
    pi0mass = np.where(pi0mass > 100, -1000, pi0mass)
    return pi0mass

def getPhi(vec1):
    # azimuthal angle of one 3d vector
    return 180/np.pi*np.arctan2(vec1[1], vec1[0])

def getTheta(vec1):
    # polar angle of one 3d vector
    return 180/np.pi*np.arctan2(np.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]), vec1[2])

def getEnergy(vec1, mass):
    # for taken 3d momenta p and mass m, return energy = sqrt(p**2 + m**2)
    return np.sqrt(mag2(vec1)+mass**2)

def readFile(fname):
    #read root using uproot
    ffile = uproot.open(fname)
    tree = ffile["T"]
    return tree
