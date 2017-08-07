import numpy as np
from keras.layers import Conv2D, Conv1D, Dropout, Flatten
from keras.layers import Dense, LeakyReLU
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt


def cnn_1d(train_input, train_output):
    n_channels = int(train_input[0].shape[1])
    n_samples = int(train_input[0].shape[0])
    train_output = to_categorical(train_output)

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
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(train_input, train_output, epochs=50)
    score = model.evaluate(train_input, train_output, batch_size=32)
    print(score)

