import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Dropout
from keras.layers.core import Flatten, Activation, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utils import image_data_generator, validation_set_generator
from utils import process_image, make_df


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
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=64, nb_row=3, nb_col=3,
        border_mode='valid',
        init='he_normal'))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=64, nb_row=3, nb_col=3,
        border_mode='valid',
        init='he_normal'))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(100, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(50, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(10, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1, init='he_normal'))

    model.summary()

    return model


def train(model, df, n_epochs=1, batch_size=256):
    """Model training function"""
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=1e-3))

    n_validation_samples = 2500
    idx = np.arange(df.shape[0], dtype=np.uint)
    np.random.shuffle(idx)
    # Any indices before this index belong to the training set
    df_train = df.iloc[idx[:-n_validation_samples]].copy()
    df_valid = df.iloc[idx[-n_validation_samples:]].copy()
    print('Number of total samples:', df.shape[0])
    print('Number of training samples:', df_train.shape[0])
    print('Number of validation samples:', df_valid.shape[0])

    filepath = 'weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]

    model.fit_generator(
        generator=image_data_generator(df=df_train,
                                       batch_size=250,
                                       shuffle=True),
        samples_per_epoch=30000,
        validation_data=validation_set_generator(df=df_valid, batch_size=250),
        nb_val_samples=n_validation_samples,
        nb_epoch=n_epochs,
        callbacks=callbacks_list,
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
    train(model=model, df=df, n_epochs=10)
