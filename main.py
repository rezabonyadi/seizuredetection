from epilepsydataprovider import SeizureDataRead
import RawDataModels
import numpy as np

kaggle_address = "R:/projects/Seizure/Seizure_detection"
folders = ["Dog_1"]
data = SeizureDataRead.read_kaggle_2014(kaggle_address, folders[0], 0.8)
n_instances = len(data["labeled_data"])
n_channels = data["labeled_data"][0].shape[0]
n_samples = data["labeled_data"][0].shape[1]


train_in = np.reshape(np.transpose(data["labeled_data"], axes=(0, 2, 1)), (n_instances, n_samples, n_channels))
# train_out = np.reshape(data["labels"], (n_instances))
RawDataModels.cnn_1d(train_in, data["labels"])


i = 0