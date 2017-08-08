import numpy as np
from keras.layers import Conv2D, Conv1D, Dropout, Flatten
from keras.layers import Dense, LeakyReLU
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, auc
import keras.backend as K


def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def cnn_1d(train_in, train_out, test_in, test_out):
    n_channels = int(train_in[0].shape[1])
    n_samples = int(train_in[0].shape[0])
    train_out = to_categorical(train_out)
    test_out = to_categorical(test_out)

    model = Sequential()
    model.add(Conv1D(n_channels * 2, 10, input_shape=(n_samples, n_channels)))
    model.add(Conv1D(n_channels * 2, 10, activation='tanh'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv1D(n_channels * 2, 5, activation='tanh'))
    model.add(Conv1D(n_channels * 2, 10))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(train_in, train_out, epochs=200, verbose=0)
    # score = model.evaluate(test_in, test_out, batch_size=32,verbose=0)
    # print("")
    # print(score)
    train_scores = model.predict_proba(train_in, verbose=0)
    test_scores = model.predict_proba(test_in, verbose=0)
    # print("")
    auc_test = roc_auc(test_out[:, 0], test_scores[:, 0])
    auc_train = roc_auc(train_out[:, 0], train_scores[:, 0])
    print('AUC test %f AUC train %f' %(auc_test, auc_train))
    return model, auc_train, auc_test

