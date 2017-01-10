from keras.models import Sequential
from keras.layers import Dense, Convolution2D
from keras.layers.core import Flatten, Activation, Lambda
from keras.optimizers import Adam
from utils import imageGenerator, process_image, parse_csv
import cv2

img = cv2.imread('data/IMG/left_2016_12_01_13_37_41_968.jpg')
img = process_image(img)
image_shape = img.shape

print('Image shape', image_shape)


def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=image_shape))
    model.add(Convolution2D(
        nb_filter=24, nb_row=5, nb_col=5,
        subsample=(2, 2),
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=36, nb_row=5, nb_col=5,
        subsample=(2, 2),
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=48, nb_row=5, nb_col=5,
        subsample=(2, 2),
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
        optimizer=Adam(lr=1e-3))

    img_file, steer = parse_csv('data/driving_log.csv', n_trim=500, delta_steering=0.08)

    history = model.fit_generator(
        generator=imageGenerator(img_file, steer, NBatchSize=256, BShuffle=True),
        samples_per_epoch=len(steer),
        nb_epoch=n_epochs,
        verbose=1)
    print('Training done')

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    model = nvidia()
    train(model=model, file_name='data/driving_log.csv', n_epochs=5)
