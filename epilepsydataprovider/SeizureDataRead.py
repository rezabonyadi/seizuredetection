import signalprocessingbank
import scipy.io as sio
import os
from epilepsydataprovider.KaggleDetection2014 import KaggleDetection2014
import numpy as np

def read_chbmit():
    return None


def read_kaggle_2014(data_address, subject, sampling_rate=-1, lat_cut=15.0):
    if subject == 'all':
        pass
        # Read all data
    else:
        # Read the data for the given participant
        data = KaggleDetection2014.read_data(data_address, subject, sampling_rate, lat_cut)
        return data

    return None

def read_freiburg():
    return None

def prepare_data(data):
    n_instances = len(data["labeled_data"])
    n_channels = data["labeled_data"][0].shape[0]
    n_samples = data["labeled_data"][0].shape[1]

    train_in = np.reshape(np.transpose(data["labeled_data"], axes=(0, 2, 1)), (n_instances, n_samples, n_channels))
    train_out = data["labels"]
    train_out_lat = data["latencies"]

    extra_instances = 10
    num_segs = 5
    sample_in_segs = int(np.floor(n_samples/num_segs))
    last_indx = sample_in_segs * num_segs
    for i in range(extra_instances):
        indx = np.random.random_integers(0, len(train_out))
        case = train_in[indx]
        last_seg = case[last_indx:, :]
        case_segs = np.reshape(case, (num_segs, sample_in_segs, n_channels))
        order = np.random.permutation(num_segs)
        case_segs_reordered = case_segs[order, ]
        case_reordered = np.reshape(case_segs_reordered, (num_segs * sample_in_segs, n_channels))
        case_reordered = np.concatenate((case_reordered, last_seg), axis=0)
        n_instances += 1
        train_in[len(train_in)] = case_reordered
        train_out.append(1)

    test_instances = len(data["unlabeled_data"])
    test_in = np.reshape(np.transpose(data["unlabeled_data"], axes=(0, 2, 1)), (test_instances, n_samples, n_channels))
    test_out = data["unlabeled_key"]
    test_out_lat = data["unlabeled_lat"]

    print("seizure/non-seizure ratio: %f" % (sum(train_out)/len(train_out)))
    return train_in, train_out, train_out_lat, test_in, test_out, test_out_lat


