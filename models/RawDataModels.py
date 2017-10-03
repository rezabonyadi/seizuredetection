import numpy as np
from keras.layers import Conv2D, Conv1D, Dropout, Flatten
from keras.layers import Dense, LeakyReLU
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.utils import plot_model
import keras.optimizers as optimizers
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
from models import KerasCallbacks

from matplotlib import pyplot as plt
import tensorflow as tf


def roc_auc(y_true, y_pred):
    # return K.tf.metrics.auc(y_true, y_pred)[0]
    # return tf.metrics.auc(y_true, y_pred)
    return roc_auc_score(y_true, y_pred)


# def my_print_fcn(x):
    # print(x)


def cnn_1d(train_in, train_out, validation_in=None, validation_out=None):
    n_channels = int(train_in[0].shape[1])
    n_samples = int(train_in[0].shape[0])
    train_out = to_categorical(train_out)
    if validation_in is not None:
        validation_out = to_categorical(validation_out)

    model = Sequential()
    model.add(Conv1D(n_channels * 2, 20, input_shape=(n_samples, n_channels)))
    model.add(Conv1D(n_channels * 2, 10, activation='tanh'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv1D(n_channels * 2, 5, activation='tanh'))
    model.add(Conv1D(n_channels * 2, 10))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # model.summary(print_fn=my_print_fcn)
    early_call_back = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    optimiser = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=1.)
    # optimiser = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=["accuracy"])

    model.fit(train_in, train_out, epochs=300, verbose=2, validation_data=(validation_in, validation_out),
              callbacks=None)

    train_scores = model.predict_proba(train_in, verbose=0)
    auc_train = roc_auc(train_out[:, 0], train_scores[:, 0])
    print('Final results (latest model): AUC train %f' % auc_train)

    # Retrive the best model and test it
    # model.set_weights(keras_recorder.best_weights)
    # train_scores = model.predict_proba(train_in, verbose=0)
    # test_scores = model.predict_proba(test_in, verbose=0)
    # auc_test = roc_auc(test_out[:, 0], test_scores[:, 0])
    # auc_train = roc_auc(train_out[:, 0], train_scores[:, 0])
    # print('Final results (best model): AUC test %f AUC train %f' %(auc_test, auc_train))
    # print('*******************************************************************')

    return model, auc_train


def cnn_2d(train_in, train_out, test_in, test_out):
    n_instances_train = len(train_in)
    n_instances_test = len(test_in)
    n_channels = int(train_in[0].shape[1])
    n_samples = int(train_in[0].shape[0])
    train_out = to_categorical(train_out)
    test_out = to_categorical(test_out)

    train_in = np.reshape(train_in, (n_instances_train, n_samples, n_channels, 1))
    test_in = np.reshape(test_in, (n_instances_test, n_samples, n_channels, 1))

    model = Sequential()
    model.add(Conv2D(n_channels * 2, (10, n_channels), input_shape=(n_samples, n_channels, 1)))
    model.add(Reshape((-1, n_channels * 2, 1)))

    # model.add(Conv2D(n_channels * 2, (10, n_channels * 2), activation='tanh'))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Reshape((-1, n_channels * 2, 1)))

    model.add(Conv2D(n_channels * 2, (10, n_channels * 2), activation='tanh'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Reshape((-1, n_channels * 2)))

    model.add(Conv1D(n_channels * 2, 5, activation='tanh'))
    model.add(Conv1D(n_channels * 2, 10))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # model.summary(print_fn=my_print_fcn)

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(train_in, train_out, epochs=1000, verbose=2)
    # score = model.evaluate(test_in, test_out, batch_size=32,verbose=0)
    # print("")
    # print(score)
    train_scores = model.predict_proba(train_in, verbose=0)
    test_scores = model.predict_proba(test_in, verbose=0)
    # print("")
    auc_test = roc_auc(test_out[:, 0], test_scores[:, 0])
    auc_train = roc_auc(train_out[:, 0], train_scores[:, 0])
    print('Final results: AUC test %f AUC train %f' %(auc_test, auc_train))
    print('*******************************************************************')
    return model, auc_train, auc_test


def evaluate_model(model, test_in, test_out):
    # Test the latest model:
    test_scores = model.predict_proba(test_in, verbose=0)
    test_out = to_categorical(test_out)

    auc_test = roc_auc(test_out[:, 0], test_scores[:, 0])
    print('Final results (latest model): AUC test %f' % auc_test)
    return auc_test
