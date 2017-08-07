import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample


class KaggleDetection2014:

    @staticmethod
    def read_data(address, sampling_rate, folder):
        ictal = "ictal"
        interictal = "interictal"
        test = "test"
        labeled_data, latencies, sequences, freq, channels = \
            KaggleDetection2014.read_data_type(address, folder, ictal, sampling_rate)
        temp_data, temp_late, temp_sequences, _, _ = \
            KaggleDetection2014.read_data_type(address, folder, interictal, sampling_rate)

        labels = np.concatenate((np.ones(len(labeled_data)), np.zeros(len(temp_data))))
        latencies.extend(temp_late)
        labeled_data.extend(temp_data)
        sequences.extend(temp_sequences)
        unlabeled_data = []
        # unlabeled_data, _, _, _, _ = KaggleDetection2014.read_data_type(address, folder, test, sampling_rate)

        data_info = dict()
        data_info["labeled_data"] = labeled_data
        data_info["labels"] = labels
        data_info["latencies_data"] = latencies
        data_info["seq"] = sequences
        data_info["unlabeled_data"] = unlabeled_data
        data_info["freq"] = freq
        data_info["channels"] = channels

        return data_info

    @staticmethod
    def read_data_type(address, folder, components, sampling_rate):
        done = False
        i = 0
        data_list = []
        latencies = []
        sequences = []
        freq = 0
        channels = []
        p_lat = 100000
        seq = 0
        print('Reading patient %s, %s ...' % (folder, components))

        while not done:
            i += 1
            filename = '%s/%s/%s_%s_segment_%d.mat' % (address, folder, folder, components, i)

            if os.path.exists(filename):
                data = sio.loadmat(filename)
                freq = data["freq"]
                channels = data["channels"]
                if components == "ictal":
                    lat = data["latency"][0]
                    if lat < p_lat:
                        seq += 1
                    p_lat = lat
                else:
                    lat = -1
                    seq = -1
                resampled_data, freq = KaggleDetection2014.resample_signal(data["data"], freq, sampling_rate)
                data_list.append(resampled_data)
                latencies.append(lat)
                sequences.append(seq)
            else:
                done = True

        return data_list, latencies, sequences, freq, channels

    @staticmethod
    def resample_signal(data, freq, sampling_rate, axis=1):
        new_freq = int(freq * sampling_rate)
        if sampling_rate == 1:
            return data, new_freq

        signal_len = data.shape[1]
        new_len = int(signal_len * sampling_rate)
        resampled = resample(data, new_len, axis=axis)
        return resampled, new_freq
