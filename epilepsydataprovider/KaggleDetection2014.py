import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample
import csv


class KaggleDetection2014:

    @staticmethod
    def read_data(address, subject, sampling_rate, latency_cut=15.0):
        ictal = "ictal"
        interictal = "interictal"
        test = "test"
        print('Reading patient %s ...' % subject)

        labeled_data, latencies, labels, sequences, freq, channels = \
            KaggleDetection2014.read_data_type(address, subject, ictal, sampling_rate, latency_cut)
        temp_data, temp_late, temp_labels, temp_sequences, _, _ = \
            KaggleDetection2014.read_data_type(address, subject, interictal, sampling_rate, latency_cut)

        # labels = np.concatenate((np.ones(len(labeled_data)), np.zeros(len(temp_data))))
        latencies.extend(temp_late)
        labels.extend(temp_labels)
        labeled_data.extend(temp_data)
        sequences.extend(temp_sequences)

        # unlabeled_data = []
        unlabeled_data, test_latencies, test_labels, _, _, _ = \
            KaggleDetection2014.read_data_type(address, subject, test, sampling_rate, latency_cut)

        data_info = dict()
        data_info["labeled_data"] = labeled_data
        data_info["labels"] = labels
        data_info["latencies"] = latencies
        data_info["seq"] = sequences
        data_info["unlabeled_data"] = unlabeled_data
        data_info["unlabeled_key"] = test_labels
        data_info["unlabeled_lat"] = test_latencies
        data_info["freq"] = freq
        data_info["channels"] = channels

        return data_info

    @staticmethod
    def read_data_type(address, folder, components, sampling_rate, latency_cut=15.0):
        done = False
        i = 0
        data_list = []
        latencies = []
        sequences = []
        freq = 0
        channels = []
        p_lat = 100000
        seq = 0

        keys_dic = dict()
        if components == 'test':
            answer_key = '%s/seizureDetectionAnswerKey.csv' % address
            with open(answer_key) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    # print(row)
                    keys_dic[row[0]] = (row[1], row[2])

        # print('Reading patient %s, %s ...' % (folder, components))
        labels = []
        while not done:
            i += 1
            file_name = '%s_%s_segment_%d.mat' % (folder, components, i)
            file_address = '%s/%s/%s_%s_segment_%d.mat' % (address, folder, folder, components, i)
            if os.path.exists(file_address):
                data = sio.loadmat(file_address)
                freq = data["freq"]
                channels = data["channels"]
                lat = 0
                if components == "ictal":
                    labels.append(1)
                    latency = data["latency"][0]
                    if latency < p_lat:
                        seq += 1
                    p_lat = latency
                    lat = int(1*(latency <= latency_cut))
                if components == "interictal":
                    lat = 0
                    seq = -1
                    labels.append(0)
                if components == "test":
                    lat = int(keys_dic[file_name][1])
                    seq = -1
                    labels.append(1*(int(keys_dic[file_name][0]) == 1))

                if sampling_rate <= 1:
                    resampled_data, freq = KaggleDetection2014.resample_signal(data["data"], freq,
                                                                               new_rate=sampling_rate)
                else:
                    resampled_data, freq = KaggleDetection2014.resample_signal(data["data"], freq,
                                                                               new_freq=sampling_rate)
                data_list.append(resampled_data)
                latencies.append(lat)
                sequences.append(seq)
            else:
                done = True

        return data_list, latencies, labels, sequences, freq, channels

    @staticmethod
    def resample_signal(data, freq, new_freq=None, new_rate=None, axis=1):
        if new_rate is None:
            sampling_rate = new_freq/freq
        else:
            sampling_rate = new_rate

        if sampling_rate == 1:
            return data, new_freq

        new_freq = int(freq * sampling_rate)

        signal_len = data.shape[1]
        new_len = int(signal_len * sampling_rate)
        resampled = resample(data, new_len, axis=axis)
        return resampled, new_freq
