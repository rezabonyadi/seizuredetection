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
    n_seizure = sum(data["labels"])
    n_non_seizure = len(data["labels"]) - n_seizure
    labeled_data, labels = expand_instances(data["labeled_data"], data["labels"], 1, n_non_seizure - n_seizure)

    n_instances = len(labeled_data)
    n_channels = labeled_data[0].shape[0]
    n_samples = labeled_data[0].shape[1]

    train_in = np.reshape(np.transpose(labeled_data, axes=(0, 2, 1)), (n_instances, n_samples, n_channels))
    train_out = labels
    train_out_lat = data["latencies"]

    test_instances = len(data["unlabeled_data"])
    test_in = np.reshape(np.transpose(data["unlabeled_data"], axes=(0, 2, 1)), (test_instances, n_samples, n_channels))
    test_out = data["unlabeled_key"]
    test_out_lat = data["unlabeled_lat"]

    print("seizure/non-seizure ratio: %f" % (sum(train_out)/len(train_out)))
    return train_in, train_out, train_out_lat, test_in, test_out, test_out_lat


def expand_instances(train_in, train_out, class_to_expand, extra_instances):
    n_instances = len(train_in)
    n_channels = train_in[0].shape[0]
    n_samples = train_in[0].shape[1]

    to_expand_indx = []

    for i in range(len(train_out)):
        if train_out[i] == class_to_expand:
            to_expand_indx.append(i)

    num_segs = 5
    sample_in_segs = int(np.floor(n_samples/num_segs))
    last_indx = sample_in_segs * num_segs
    for i in range(extra_instances):
        indx = np.random.random_integers(0, len(to_expand_indx) - 1)
        indx = to_expand_indx[indx] # True index
        instance_to_perturb = train_in[indx]
        last_seg = instance_to_perturb[last_indx:, :]
        perturbed_segs = np.reshape(instance_to_perturb, (n_channels, num_segs, sample_in_segs))
        order = np.random.permutation(num_segs)
        perturbed_segs_reordered = perturbed_segs[:, order, ]
        perturbed_instance_reordered = np.reshape(perturbed_segs_reordered, (n_channels, num_segs * sample_in_segs))
        perturbed_instance_reordered = np.concatenate((perturbed_instance_reordered, last_seg), axis=0)
        train_in.append(perturbed_instance_reordered)
        train_out.append(class_to_expand)

    return train_in, train_out
