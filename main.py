from epilepsydataprovider import SeizureDataRead
import RawDataModels
import numpy as np
import json
import csv

with open('SETTINGS.json') as f:
    settings = json.load(f)

kaggle_address = str(settings['kaggle-data-dir'])

folders = ["Dog_1", "Dog_2"]#, "Dog_3", "Dog_4", "Patient_1", "Patient_2", "Patient_3", "Patient_4", "Patient_5",
           #"Patient_6", "Patient_7", "Patient_8"]
# folders = ["Patient_1"]
model_details = dict()

for subject in folders:
    data = SeizureDataRead.read_kaggle_2014(kaggle_address, subject, .3)
    print("Modeling of %s in progress" %subject)
    train_in, train_out, train_lat, test_in, test_out, test_lat = SeizureDataRead.prepare_data(data)
    # train_out = np.reshape(data["labels"], (n_instances))
    model, acc_train, acc_test = RawDataModels.cnn_1d(train_in, train_out, test_in, test_out)
    model_details[subject + "-model"] = model.to_json()
    model_details[subject + "-auc_train"] = acc_train
    model_details[subject + "-auc_test"] = acc_test

with open('result.json', 'w') as fp:
    json.dump(model_details, fp, indent=4)

i = 0