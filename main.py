from epilepsydataprovider import SeizureDataRead
import RawDataModels
import numpy as np

kaggle_address = "R:/projects/Seizure/Seizure_detection"
folders = ["Dog_1"]
data = SeizureDataRead.read_kaggle_2014(kaggle_address, folders[0], 0.8)



# train_out = np.reshape(data["labels"], (n_instances))
# RawDataModels.cnn_1d(train_in, data["labels"])


i = 0