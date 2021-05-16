from utils.utilities import split_data
from utils.utilities import cartesian_converter
import pandas as pd
import numpy as np
import pickle5 as pickle

if __name__ == "__main__":
    with open('data/pi0.pkl', 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float64)
    x = cartesian_converter(xz) #pi0.pkl is in spherical coordinates, need to convert to cartesian
    dfx = pd.DataFrame(x)
    train,test = split_data(dfx)

    train.to_pickle("data/pi0_cartesian_train.pkl")
    test.to_pickle("data/pi0_cartesian_test.pkl")

