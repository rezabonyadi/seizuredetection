import signalprocessingbank
import scipy.io as sio
import os
from epilepsydataprovider.KaggleDetection2014 import KaggleDetection2014
import numpy as np

def read_chbmit():
    return None


def read_kaggle_2014(data_address, folder, sampling_rate=-1):
    if folder == 'all':
        pass
        # Read all data
    else:
        # Read the data for the given participant
        data = KaggleDetection2014.read_data(data_address, sampling_rate, folder)

        return data

    return None

def read_freiburg():
    return None

def prepare_data(data, latency_cut=15.0):
    n_instances = len(data["labeled_data"])
    n_channels = data["labeled_data"][0].shape[0]
    n_samples = data["labeled_data"][0].shape[1]

    train_in = np.reshape(np.transpose(data["labeled_data"], axes=(0, 2, 1)), (n_instances, n_samples, n_channels))
    train_out = data["labels"]
    train_in_lat = []
    train_out_lat = []

    test_in = []
    test_out = []
    test_in_lat = []
    test_out_lat = []

    test_in = np.reshape(np.transpose(data["unlabeled_data"], axes=(0, 2, 1)),
                                      (n_instances, n_samples, n_channels))



