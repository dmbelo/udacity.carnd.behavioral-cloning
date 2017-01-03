from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Activation, Lambda
import numpy as np

image_shape = (66, 200, 3)

def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=image_shape))
    model.add(Convolution2D(
        nb_filter=24, nb_row=5, nb_col=5,
        subsample=(2,2),
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=36, nb_row=5, nb_col=5,
        subsample=(2,2),
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=48, nb_row=5, nb_col=5,
        subsample=(2,2),
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=64, nb_row=3, nb_col=3,
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=64, nb_row=3, nb_col=3,
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(100, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(50, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(10, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(1, init='he_normal'))

    model.summary()

    return model

def train(model, n_epochs, batch_size, xTrain, yTrain, xVal, yVal):

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=2e-3))

    history = model.fit(
        xTrain,
        yTrain,
        batch_size=batch_size,
        nb_epoch=n_epochs,
        verbose=1,
        validation_data=(xVal, yVal))

    score = model.evaluate(xVal, yVal, verbose=1)

if __name__ == "__main__":
    model = nvidia()
