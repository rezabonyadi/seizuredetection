import signalprocessingbank
import scipy.io as sio
import os
from epilepsydataprovider.KaggleDetection2014 import KaggleDetection2014
import numpy as np
from sklearn import preprocessing
import random


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


def prepare_data(data, processes):
    transformation = processes["transform"]
    normalise = processes["normalise"]
    class_to_expand = processes["expand"]
    val_percentage = processes["val_percentage"]

    n_seizure = sum(data["labels"])
    n_non_seizure = len(data["labels"]) - n_seizure
    labeled_data = data["labeled_data"]
    labels = data["labels"]
    latencies = data["latencies"]
    unlabeled_data = data["unlabeled_data"]
    seqs = data["seq"]

    labeled_data, labels, seqs, latencies = expand_instances(labeled_data, labels, latencies, seqs,
                                                             class_to_expand, n_non_seizure - n_seizure)

    labeled_data = transform(labeled_data, transformation, normalise)
    unlabeled_data = transform(unlabeled_data, transformation, normalise)

    n_channels = labeled_data[0].shape[0]
    n_samples = labeled_data[0].shape[1]

    if val_percentage is not 0:
        vals, val_lab, val_lat, val_seqs = \
            divide_to_val(labeled_data, labels, latencies, seqs, perc=val_percentage)
        val_instances = len(vals)
        vals = np.reshape(np.transpose(vals, axes=(0, 2, 1)), (val_instances, n_samples, n_channels))
    else:
        vals = None
        val_lab = None
        val_lat = None
        val_seqs = None

    train_instances = len(labeled_data)
    labeled_data = np.reshape(np.transpose(labeled_data, axes=(0, 2, 1)), (train_instances, n_samples, n_channels))

    test_instances = len(data["unlabeled_data"])
    unlabeled_data = np.reshape(np.transpose(unlabeled_data, axes=(0, 2, 1)), (test_instances, n_samples, n_channels))
    test_out = data["unlabeled_key"]
    test_out_lat = data["unlabeled_lat"]

    print("seizure/non-seizure ratio: %f" % (sum(labels)/len(labels)))
    return labeled_data, labels, latencies, seqs, vals, val_lab, val_lat, val_seqs, unlabeled_data, test_out, test_out_lat


def divide_to_val(data, labels, latencies, seqs, perc=0.25):
    # labeled_data = data
    # labels = labels
    # train_out_lat = latencies
    unique_seqs = sorted(list(set(seqs)))
    unique_seqs.pop(0) # Ignore -1 as it is the sequence number for interictals

    num_of_val = (int)(np.floor(perc*(len(unique_seqs))))
    num_of_val = max(num_of_val, 1)
    val_indces = random.sample(unique_seqs, num_of_val)
    all_seq_indx = []

    for val_indx in val_indces:
        all_seq_indx.extend(list(np.where(np.array(seqs) == val_indx)[0]))

    interIctals = list(np.where(np.array(seqs) == -1)[0])
    num_of_val = (int)(np.floor(perc * (len(interIctals))))
    val_indces = random.sample(interIctals, num_of_val)
    all_seq_indx.extend(val_indces)

    vals = []
    val_lab = []
    val_lat = []
    val_seq = []

    for indx in sorted(all_seq_indx, reverse=True):
        vals.append(data[indx])
        val_lab.append(labels[indx])
        val_lat.append(latencies[indx])
        val_seq.append(seqs[indx])
        del (data[indx])
        del (labels[indx])
        del (latencies[indx])
        del (seqs[indx])

    # get_valid_data(data, labels, latencies, seqs, all_seq_indx, val_lab, val_lat, val_seq, vals)

    return vals, val_lab, val_lat, val_seq


# def get_valid_data(data, labels, latencies, seqs, all_seq_indx, val_lab, val_lat, val_seq, vals):

def expand_instances(train_in, train_out, latencies, seqs, class_to_expand, num_extra_instances):
    if class_to_expand is None:
        return train_in, train_out, seqs, latencies

    n_instances = len(train_in)
    n_channels = train_in[0].shape[0]
    n_samples = train_in[0].shape[1]

    to_expand_indx = []

    for i in range(len(train_out)):
        if train_out[i] == class_to_expand:
            to_expand_indx.append(i)

    # segement_augment(class_to_expand, latencies, n_channels, n_samples, num_extra_instances, seqs, to_expand_indx,
    #                  train_in, train_out)
    rotation_augment(class_to_expand, latencies, n_channels, n_samples, num_extra_instances, seqs, to_expand_indx,
                     train_in, train_out)

    return train_in, train_out, seqs, latencies


def segement_augment(class_to_expand, latencies, n_channels, n_samples, num_extra_instances, seqs, to_expand_indx,
                     train_in, train_out):
    num_segs = 5
    sample_in_segs = int(np.floor(n_samples / num_segs))
    last_indx = sample_in_segs * num_segs
    for i in range(num_extra_instances):
        indx = np.random.random_integers(0, len(to_expand_indx) - 1)
        indx = to_expand_indx[indx]  # True index
        instance_to_perturb = train_in[indx]
        last_seg = instance_to_perturb[last_indx:, :]
        perturbed_segs = np.reshape(instance_to_perturb, (n_channels, num_segs, sample_in_segs))
        order = np.random.permutation(num_segs)
        perturbed_segs_reordered = perturbed_segs[:, order, ]
        perturbed_instance_reordered = np.reshape(perturbed_segs_reordered, (n_channels, num_segs * sample_in_segs))
        perturbed_instance_reordered = np.concatenate((perturbed_instance_reordered, last_seg), axis=0)
        seqs.append(seqs[indx])
        latencies.append(latencies[indx])
        train_in.append(perturbed_instance_reordered)
        train_out.append(class_to_expand)


def rotation_augment(class_to_expand, latencies, n_channels, n_samples, num_extra_instances, seqs, to_expand_indx,
                     train_in, train_out):

    for i in range(num_extra_instances):
        indx = np.random.random_integers(0, len(to_expand_indx) - 1)
        indx = to_expand_indx[indx]  # True index
        instance_to_perturb = train_in[indx]

        M = random_rotation(n_channels, 2)
        perturbed_instance_reordered = np.dot(M, instance_to_perturb)

        seqs.append(seqs[indx])
        latencies.append(latencies[indx])
        train_in.append(perturbed_instance_reordered)
        train_out.append(class_to_expand)


def random_rotation(n, order):
    W = np.random.rand(n, n) - 0.5
    W = W - np.transpose(W)
    theta = np.random.rand(1) * 10  # 10 degrees random
    Wp = (theta*np.pi/180.0) * W
    Wpi = np.identity(n)
    M = np.identity(n)
    for j in range(order):
        Wpi = np.dot(Wpi, Wp)
        M = M + Wpi

    return M


def transform(data, transformation, normalise):
    if transformation == 'fft':
        n_samples = data[0].shape[1]
        fft_data = np.absolute(np.fft.fft(data, n=None, axis=2))
        fft_data = fft_data[:, :, 0:int(n_samples / 2)]
        # normalized_d = normalisation(fft_data, normalise)
        normalized_d = []
        for d in fft_data:
            temp = np.apply_along_axis(np.divide, 0, d, np.sum(d, axis=1))
            normalized_d.append(temp)

    else:
        normalized_d = normalisation(data, normalise)

    return normalized_d


def normalisation(data, normalise):
    if normalise is not None:
        normalised_data = []
        for d in data:
            normed = preprocessing.scale(d, axis=normalise)
            normalised_data.append(normed)
        return normalised_data
    return data
