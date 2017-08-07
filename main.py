from epilepsydataprovider import SeizureDataRead
import RawDataModels
import numpy as np
import json


with open('SETTINGS.json') as f:
    settings = json.load(f)

kaggle_address = str(settings['kaggle-data-dir'])

folders = ["Dog_1"]
data = SeizureDataRead.read_kaggle_2014(kaggle_address, folders[0], 0.8)
train_in, train_out = SeizureDataRead.prepare_data(data)
# train_out = np.reshape(data["labels"], (n_instances))
RawDataModels.cnn_1d(train_in, train_out)


i = 0