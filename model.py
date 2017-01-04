from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Activation, Lambda
from keras.optimizers import Adam
import numpy as np
from utils import imageGenerator

# image_shape = (66, 200, 3)
image_shape = (160, 320, 3)

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

def train(model, file_name, n_epochs=1, batch_size=256):

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=2e-3))

    history = model.fit_generator(
        generator=imageGenerator(file_name, NBatchSize=256, BShuffle=True),
        samples_per_epoch=8036,
        nb_epoch=n_epochs,
        verbose=1)

if __name__ == "__main__":
    model = nvidia()
    train(model=model, file_name='data/driving_log.csv', n_epochs=5)
    print('Training done')
