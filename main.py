from epilepsydataprovider import SeizureDataRead
import RawDataModels
import os
from time import strftime, localtime
import numpy as np
import json
import csv


def save_details(model_details, res_details, folder):
    if not os.path.exists('results/' + folder):
        os.makedirs('results/' + folder)

    c_time = strftime('%Y-%m-%d-%H-%M-%S', localtime())
    f_name = 'results/%s/models-%s.json' % (folder, c_time)

    with open(f_name, 'w') as fp:
        json.dump(model_details, fp, indent=4)

    f_name = 'results/%s/res-%s.csv' % (folder, c_time)

    with open(f_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('subject', 'auc_train', 'auc_test'))
        [writer.writerow(r) for r in res_details]


def kaggle_data_2014_patient_specific(address):
    folders = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Patient_1", "Patient_2", "Patient_3", "Patient_4", "Patient_5",
               "Patient_6", "Patient_7", "Patient_8"]
    # folders = ["Dog_2"]
    model_details = dict()

    res_details = []

    for subject in folders:
        data = SeizureDataRead.read_kaggle_2014(address, subject, 1.0)
        print("Modeling of %s in progress" %subject)
        train_in, train_out, train_lat, test_in, test_out, test_lat = SeizureDataRead.prepare_data(data)
        # train_out = np.reshape(data["labels"], (n_instances))
        # model, acc_train, acc_test = RawDataModels.cnn_1d(train_in, train_out, test_in, test_out)
        model, acc_train, acc_test = RawDataModels.cnn_2d(train_in, train_out, test_in, test_out)
        model_details[subject] = model.to_json()

        res_details.append((subject, acc_train, acc_test))

    save_details(model_details, res_details, 'kaggle_2014')


with open('SETTINGS.json') as f:
    settings = json.load(f)
kaggle_address = str(settings['kaggle-data-dir'])
if not os.path.exists('results'):
    os.makedirs('results')

kaggle_data_2014_patient_specific(kaggle_address)

i = 0