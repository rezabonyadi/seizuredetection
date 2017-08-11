import numpy as np
from keras.layers import Conv2D, Conv1D, Dropout, Flatten
from keras.layers import Dense, LeakyReLU
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.utils import plot_model
import keras.callbacks as KCallBacks
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import keras.backend as K
import tensorflow as tf


class roc_callback(KCallBacks.Callback):
    def __init__(self, train_in, train_out, val_in, val_out):
        self.x = train_in
        self.y = train_out
        self.x_val = val_in
        self.y_val = val_out


    def on_train_begin(self, logs=None):
        self.best_auc = -1.0
        self.best_weights = self.model.get_weights()
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict_proba(self.x, verbose=0)
        auc = roc_auc_score(self.y[:, 0], y_pred[:, 0])

        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)
        auc_val = roc_auc_score(self.y_val[:, 0], y_pred_val[:, 0])

        total_auc = (auc_val + auc) / 2
        if total_auc > self.best_auc:
            self.best_auc = total_auc
            self.best_weights = self.model.get_weights()
            # print("Best updated to: %f" % self.best_auc)

        # print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(auc, 4)), str(round(auc_val, 4))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return

class KerasRecorder(KCallBacks.Callback):
    def on_train_begin(self, logs=None):
        self.best_loss = 1000000
        self.best_weights = self.model.get_weights()

    # def on_batch_end(self, batch, logs=None):
    #     self.loss.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        loss = 0 #logs['loss']
        total_loss = (val_loss + loss)/2
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_weights = self.model.get_weights()
            # print("Best updated to: %f" %self.best_loss)

def roc_auc(y_true, y_pred):
    # return K.tf.metrics.auc(y_true, y_pred)[0]
    # return tf.metrics.auc(y_true, y_pred)
    return roc_auc_score(y_true, y_pred)


# def my_print_fcn(x):
    # print(x)


def cnn_1d(train_in, train_out, test_in, test_out):
    n_channels = int(train_in[0].shape[1])
    n_samples = int(train_in[0].shape[0])
    train_out = to_categorical(train_out)
    test_out = to_categorical(test_out)

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

    # model = Sequential()
    # model.add(Conv1D(n_channels * 2, 10, input_shape=(n_samples, n_channels)))
    # model.add(Conv1D(n_channels * 2, 10, activation='tanh'))
    # # model.add(LeakyReLU(alpha=0.3))
    # model.add(Conv1D(n_channels * 2, 5, activation='relu'))
    # model.add(Conv1D(n_channels * 2, 10))
    # # model.add(LeakyReLU(alpha=0.3))
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='sgd')
    # model.compile(loss=roc_auc, optimizer='sgd')
    # keras_recorder = KerasRecorder()
    keras_recorder = roc_callback(train_in, train_out, test_in, test_out)

    model.fit(train_in, train_out, epochs=100, verbose=2,
               callbacks=[keras_recorder])


    # Test the latest model:
    train_scores = model.predict_proba(train_in, verbose=0)
    test_scores = model.predict_proba(test_in, verbose=0)
    auc_test = roc_auc(test_out[:, 0], test_scores[:, 0])
    auc_train = roc_auc(train_out[:, 0], train_scores[:, 0])
    print('Final results (latest model): AUC test %f AUC train %f' % (auc_test, auc_train))

    # Retrive the best model and test it
    model.set_weights(keras_recorder.best_weights)
    train_scores = model.predict_proba(train_in, verbose=0)
    test_scores = model.predict_proba(test_in, verbose=0)
    auc_test = roc_auc(test_out[:, 0], test_scores[:, 0])
    auc_train = roc_auc(train_out[:, 0], train_scores[:, 0])
    print('Final results (best model): AUC test %f AUC train %f' %(auc_test, auc_train))
    print('*******************************************************************')

    return model, auc_train, auc_test


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
    print('Final results: AUC test %f AUC train %f' %(auc_test, auc_train))
    print('*******************************************************************')
    return model, auc_train, auc_test
