import signalprocessingbank
import scipy.io as sio
import os
from epilepsydataprovider.KaggleDetection2014 import KaggleDetection2014

def read_chbmit():
    return None


def read_kaggle_2014(data_address, folder, sampling_rate=-1):
    if folder == 'all':
        pass
        # Read all data
    else:
        # Read the data for the given participant

        return KaggleDetection2014.read_data(data_address, sampling_rate, folder)

    return None

def read_freiburg():
    return None

def prepare_data(data, latency_cut=15.0):
    data_info = dict()
    data_info["freq"] = data["freq"]


