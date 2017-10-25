import numpy as np
from keras.layers import Conv2D, Conv1D, Dropout, Flatten, MaxPooling1D, Input, LSTM, MaxPooling2D
from keras.layers import Dense, LeakyReLU
from keras.layers.core import Reshape
from keras.models import Sequential, Model
from keras.utils import plot_model
import keras.optimizers as optimizers
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
from models import KerasCallbacks
from keras import backend as K
from matplotlib import pyplot as plt
import tensorflow as tf


def roc_auc(y_true, y_pred):
    # return K.tf.metrics.auc(y_true, y_pred)[0]
    # return tf.metrics.auc(y_true, y_pred)
    # K.set_image_data_format()
    return roc_auc_score(y_true, y_pred)


# def my_print_fcn(x):
    # print(x)


def cnn(train_in, train_out, model_indx, validation_in=None, validation_out=None):
    n_channels = int(train_in[0].shape[1])
    n_samples = int(train_in[0].shape[0])
    # n_instances_train = int(train_in.shape[0])

    train_out = to_categorical(train_out)
    if validation_in is not None:
        validation_out = to_categorical(validation_out)

    # train_in = np.reshape(train_in, (n_instances_train, n_samples, n_channels, 1))
    # test_in = np.reshape(test_in, (n_instances_test, n_samples, n_channels, 1))

    model = init_model(model_indx, n_channels, n_samples)
    keras_recorder = KerasCallbacks.BestRecorder()
    early_call_back = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

    showRes = 2

    if validation_in is not None:
        model.fit(train_in, train_out, epochs=300, verbose=showRes, validation_data=(validation_in, validation_out),
                  callbacks=[keras_recorder])
        test_scores = model.predict(validation_in, verbose=0)
        auc_validation = roc_auc(validation_out[:, 0], test_scores[:, 0])
    else:
        model.fit(train_in, train_out, epochs=300, verbose=showRes)
        test_scores = 0
        auc_validation = 0

    # Retrive the best model and test it
    train_scores = model.predict(train_in, verbose=0)
    auc_train = roc_auc(train_out[:, 0], train_scores[:, 0])
    print('Final results (last model): AUC validation %f AUC train %f' % (auc_validation, auc_train))
    print('*******************************************************************')

    # model.set_weights(keras_recorder.best_weights)
    # train_scores = model.predict(train_in, verbose=0)
    # test_scores = model.predict(validation_in, verbose=0)
    # auc_validation = roc_auc(validation_out[:, 0], test_scores[:, 0])
    # auc_train = roc_auc(train_out[:, 0], train_scores[:, 0])
    # print('Final results (best model): AUC validation %f AUC train %f' %(auc_validation, auc_train))
    # print('*******************************************************************')

    return model, auc_train, auc_validation


# def cnn_2d(train_in, train_out, test_in, test_out):
#     n_instances_train = len(train_in)
#     n_instances_test = len(test_in)
#     n_channels = int(train_in[0].shape[1])
#     n_samples = int(train_in[0].shape[0])
#     train_out = to_categorical(train_out)
#     test_out = to_categorical(test_out)
#
#     train_in = np.reshape(train_in, (n_instances_train, n_samples, n_channels, 1))
#     test_in = np.reshape(test_in, (n_instances_test, n_samples, n_channels, 1))
#
    # model = Sequential()
    #
    # model.add(Conv2D(n_channels * 2, (10, n_channels), input_shape=(n_samples, n_channels, 1)))
    # model.add(Reshape((-1, n_channels * 2, 1)))
    #
    # # model.add(Conv2D(n_channels * 2, (10, n_channels * 2), activation='tanh'))
    # # model.add(LeakyReLU(alpha=0.3))
    # # model.add(Reshape((-1, n_channels * 2, 1)))
    #
    # model.add(Conv2D(n_channels * 2, (10, n_channels * 2), activation='tanh'))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Reshape((-1, n_channels * 2)))
    #
    # model.add(Conv1D(n_channels * 2, 5, activation='tanh'))
    # model.add(Conv1D(n_channels * 2, 10))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='softmax'))
    # model.summary(print_fn=my_print_fcn)
#
#     model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#     model.fit(train_in, train_out, epochs=1000, verbose=2)
#     # score = model.evaluate(test_in, test_out, batch_size=32,verbose=0)
#     # print("")
#     # print(score)
#     train_scores = model.predict_proba(train_in, verbose=0)
#     test_scores = model.predict_proba(test_in, verbose=0)
#     # print("")
#     auc_test = roc_auc(test_out[:, 0], test_scores[:, 0])
#     auc_train = roc_auc(train_out[:, 0], train_scores[:, 0])
#     print('Final results: AUC test %f AUC train %f' %(auc_test, auc_train))
#     return model, auc_train, auc_test


def evaluate_model(model, test_in, test_out):
    # Test the latest model:
    test_scores = model.predict(test_in, verbose=0)
    test_out = to_categorical(test_out)

    auc_test = roc_auc(test_out[:, 0], test_scores[:, 0])
    print('Final results: AUC test %f' % auc_test)
    return auc_test


def init_model(indx, n_channels, n_samples):
    model = []
    if indx == 1:
        # model = Sequential()
        inputs = Input(shape=(n_samples, n_channels))
        cnv1 = Conv1D(n_channels * 2, 5, activation='relu', strides=1, padding='valid')(inputs)
        cnv2 = Conv1D(n_channels * 4, 5, activation='relu', padding='valid')(cnv1)
        cnv3 = Conv1D(n_channels * 4, 5, activation='relu', padding='valid')(cnv2)
        cnv4 = Conv1D(n_channels * 4, 5, activation='relu', padding='valid')(cnv3)
        flt1 = Flatten()(cnv4)
        drp1 = Dropout(0.2)(flt1)
        dns1 = Dense(64, activation='sigmoid')(drp1)
        drp2 = Dropout(0.5)(dns1)
        dns2 = Dense(2, activation='softmax')(drp2)
        model = Model(input=inputs, output=dns2)

    if indx == 2:
        inputs = Input(shape=(n_samples, n_channels))
        cnv1 = Conv1D(n_channels * 2, 4, activation='relu', strides=1, padding='valid')(inputs)
        pol1 = MaxPooling1D(pool_size=4, padding='valid')(cnv1)
        cnv2 = Conv1D(n_channels * 2, 4, activation='relu', padding='valid')(pol1)
        pol2 = MaxPooling1D(pool_size=2, padding='valid')(cnv2)
        cnv3 = Conv1D(n_channels * 4, 4, activation='relu', padding='valid')(pol2)
        pol3 = MaxPooling1D(pool_size=2, padding='valid')(cnv3)
        cnv4 = Conv1D(n_channels * 8, 4, activation='relu', padding='valid')(pol3)
        pol4 = MaxPooling1D(pool_size=2, padding='valid')(cnv4)
        flt1 = Flatten()(pol4)
        drp1 = Dropout(0.2)(flt1)
        dns1 = Dense(128, activation='sigmoid')(drp1)
        drp2 = Dropout(0.5)(dns1)
        dns2 = Dense(2, activation='softmax')(drp2)
        model = Model(input=inputs, output=dns2)

    if indx == 3:
        inputs = Input(shape=(n_samples, n_channels))
        rsh0 = Reshape((-1, n_channels, 1))(inputs)
        cnv1 = Conv2D(n_channels * 2, (4, n_channels), activation='relu', strides=(1, 1), padding='valid')(rsh0)
        pol1 = MaxPooling2D(pool_size=(2, 1), padding='valid')(cnv1)
        rsh1 = Reshape((-1, n_channels * 2, 1))(pol1)
        cnv2 = Conv2D(n_channels * 2, (4, n_channels * 2), activation='relu', strides=(1, 1), padding='valid')(rsh1)
        pol2 = MaxPooling2D(pool_size=(2, 1), padding='valid')(cnv2)
        flt1 = Flatten()(pol2)
        drp1 = Dropout(0.2)(flt1)
        dns1 = Dense(128, activation='sigmoid')(drp1)
        drp2 = Dropout(0.5)(dns1)
        dns2 = Dense(2, activation='softmax')(drp2)
        model = Model(input=inputs, output=dns2)

    # model.summary()
    optimiser = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=1.)
    # optimiser = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optimiser = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=["accuracy"])

    return model