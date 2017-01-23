from keras.models import Sequential
from keras.layers import Dense, Convolution2D
from keras.layers.core import Flatten, Activation, Lambda
from keras.optimizers import Adam
from utils import image_data_generator, process_image, make_df
import cv2


sample_image = cv2.imread('data/IMG/left_2016_12_01_13_37_41_968.jpg')
image_shape = process_image(sample_image).shape
print('Image Shape:', image_shape)


def nvidia():
    """Implementation of the NVIDIA model"""
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
        subsample=(1, 2),
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=48, nb_row=5, nb_col=5,
        subsample=(1, 2),
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


def train(model, df, n_epochs=1, batch_size=256):
    """Model training function""""
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=1e-3))

    model.fit_generator(
        generator=image_data_generator(df=df,
                                       batch_size=256,
                                       shuffle=True),
        samples_per_epoch=30000,
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
    """Run the training"""
    model = nvidia()
    df = make_df('data/driving_log.csv')
    train(model=model, df=df, n_epochs=5)
