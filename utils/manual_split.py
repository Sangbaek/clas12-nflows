from utilities import split_data
from utilities import cartesian_converter
import pandas as pd
import numpy as np
import pickle5 as pickle

if __name__ == "__main__":
    with open('data/epgg.pkl', 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float64)
    dfx = pd.DataFrame(xz)
    train,test = split_data(dfx)

    train.to_pickle("data/epgg_cartesian_train.pkl")
    test.to_pickle("data/epgg_cartesian_test.pkl")

    with open('data/pi0_cartesian.pkl', 'rb') as f2:
        xz2 = np.array(pickle.load(f2), dtype=np.float64)
    dfxz2 = pd.DataFrame(xz2)
    train,test = split_data(dfxz2)
    train.to_pickle("data/pi0_cartesian_train.pkl")
    test.to_pickle("data/pi0_cartesian_test.pkl")