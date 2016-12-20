from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Activation
import numpy as np

image_shape = (66, 200, 3)

def alpha1():
    model = Sequential()
    model.add(Convolution2D(
        nb_filter=24,
        nb_row=5,
        nb_col=5,
        subsample=(2,2),
        border_mode='valid',
        input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=36,
        nb_row=5,
        nb_col=5,
        subsample=(2,2),
        border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=48,
        nb_row=5,
        nb_col=5,
        subsample=(2,2),
        border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=64,
        nb_row=3,
        nb_col=3,
        border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=64,
        nb_row=3,
        nb_col=3,
        border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.summary()

    return model

def train(model, NEpochs=1, NBatchSize=100, xTrain, yTrain, xVal, yVal):

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=2e-3))

    history = model.fit(
        xTrain,
        yTrain,
        batch_size=NBatchSize,
        nb_epoch=NEpochs,
        verbose=1,
        validation_data=(xVal, yVal))

    score = model.evaluate(xVal, yVal, verbose=1)

if __name__ == "__main__":
    model = alpha1()
