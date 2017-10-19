import csv
import json
import os
from time import strftime, localtime
import numpy as np

from epilepsydataprovider import SeizureDataRead
from models import RawDataModels


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
    # folders = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Patient_1", "Patient_2", "Patient_3", "Patient_4", "Patient_5",
    #            "Patient_6", "Patient_7", "Patient_8"]
    # folders = ["Patient_1", "Patient_2", "Patient_3", "Patient_4", "Patient_5",
    #            "Patient_6", "Patient_7", "Patient_8"]
    folders = ["Dog_1", "Dog_2", "Dog_3", "Dog_4"]

    # folders = ["Patient_3"]

    processes = dict()
    processes["transform"] = None
    processes["normalise"] = 0  # 0 is across channels, 1 is over samples, None is no normalisation
    processes["expand"] = 1  # expands the seizure examples
    processes["samp_rate"] = 200
    model_indx = 2

    model_details = dict()
    train_details = np.zeros((len(folders), len(folders)))
    test_details = np.zeros((len(folders), len(folders)))

    i = 0
    j = 0

    for subject in folders:
        data = SeizureDataRead.read_kaggle_2014(address, subject, 100)
        print("Modeling of %s in progress" %subject)
        # train_in, train_out, train_lat, test_in, test_out, test_lat = SeizureDataRead.prepare_data(data, processes)
        train_in, train_out, train_lat, seqs, \
        val_in, val_out, val_lat, val_seqs, \
        test_in, test_out, test_lat = SeizureDataRead.prepare_data(data, processes)
        model, auc_train, auc_val = RawDataModels.cnn(train_in, train_out, model_indx, val_in, val_out)

        # model, acc_train = RawDataModels.cnn_1d(train_in, train_out, test_in, test_out)
        model_details[subject] = model.to_json()
        j = 0
        for subject_test in folders:
            print("Testing on subject %s " %subject_test)
            data = SeizureDataRead.read_kaggle_2014(address, subject_test, 100)
            train_in, train_out, train_lat, test_in, test_out, test_lat = SeizureDataRead.prepare_data(data, processes)
            acc_test = RawDataModels.evaluate_model(model, test_in, test_out)
            acc_train_test = RawDataModels.evaluate_model(model, train_in, train_out)
            train_details[i, j] = acc_train_test
            test_details[i, j] = acc_test
            j += 1
        i += 1

    i = 0
        # model, acc_train, acc_test = RawDataModels.cnn_2d(train_in, train_out, test_in, test_out)

        # res_details.append((subject, acc_train, acc_test))
    #
    # save_details(model_details, res_details, 'kaggle_2014')


with open('SETTINGS.json') as f:
    settings = json.load(f)
kaggle_address = str(settings['kaggle-data-dir'])
if not os.path.exists('results'):
    os.makedirs('results')

kaggle_data_2014_patient_specific(kaggle_address)

i = 0