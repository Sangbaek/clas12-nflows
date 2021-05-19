from utils.utilities import split_data
from utils.utilities import cartesian_converter
import pandas as pd
import numpy as np
import pickle5 as pickle

if __name__ == "__main__":
    with open('data/pi0.pkl', 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float64)

    dfxz = pd.DataFrame(xz)
    train,test = split_data(dfxz)

    train.to_pickle("data/pi0_train.pkl")
    test.to_pickle("data/pi0_test.pkl")

    with open('data/epgg.pkl', 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float64)

    dfxz = pd.DataFrame(xz)
    train,test = split_data(dfxz)

    train.to_pickle("data/epgg_train.pkl")
    test.to_pickle("data/epgg_test.pkl")
