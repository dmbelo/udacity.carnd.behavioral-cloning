import pandas as pd
import numpy as np
import cv2


def plot(df):
    """Plotting helper to visualize the udacity data set"""
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(411)
    plt.plot(df.steering)
    plt.xlabel('Index')
    plt.ylabel('Steering')

    plt.subplot(412, sharex=ax1)
    plt.plot(df.throttle)
    plt.xlabel('Index')
    plt.ylabel('Throttle')

    plt.subplot(413, sharex=ax1)
    plt.plot(df.brake)
    plt.xlabel('Index')
    plt.ylabel('Brake')

    plt.subplot(414, sharex=ax1)
    plt.plot(df.speed)
    plt.xlabel('Index')
    plt.ylabel('Speed')

    plt.show(block=False)


def play(df):
    """Helper to visualize the images in the udacity data set in a stream"""
    for i, file in enumerate(df.image_file):
        img = cv2.imread('data/' + file, 0)

        idx = df.index[i]
        cv2.putText(img,
                    'index = {:d}'.format(idx),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 180, 180))

        cv2.putText(img,
                    'speed = {:.2f}'.format(df.loc[idx, 'speed']),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 180, 180))

        cv2.putText(img,
                    'steer = {:.2f}'.format(df.loc[idx, 'steering']),
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 180, 180))

        cv2.imshow('image', img)
        if cv2.waitKey(33) == 27:
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(33) == 32:
            while True:
                if cv2.waitKey(33) == 32:
                    break

    cv2.destroyAllWindows()


def down_sample_zeros(df, zeros):
    """
    Down-sample the zero-steer data

    Retain only a random sample of zeros in pandas.DataFrame df and remove rest
    """

    b_zero_steer = df.steering == 0
    idx = b_zero_steer.index[b_zero_steer].values
    np.random.shuffle(idx)
    idx_remove_zeros = idx[zeros:]
    return df.drop(idx_remove_zeros, axis=0)


def filter_01(df):
    """Remove front, middle and ending sections of the udacity data set"""
    b_ends = (df.index < 80) | (df.index > 7790)
    b_middle = (df.index > 3400) & (df.index < 4600)
    return df.loc[(~b_ends) & (~b_middle), :].copy()


def collapse(df, steer_offset):
    """Collapse center, left and right images into one and add steer offset"""
    c = ['center', 'steering', 'throttle', 'brake', 'speed']
    df_center = df.loc[:, c].copy()
    df_center.rename(columns={'center': 'image_file'}, inplace=True)

    c = ['left', 'steering', 'throttle', 'brake', 'speed']
    df_left = df.loc[:, c].copy()
    df_left.rename(columns={'left': 'image_file'}, inplace=True)
    df_left.steering += steer_offset

    c = ['right', 'steering', 'throttle', 'brake', 'speed']
    df_right = df.loc[:, c].copy()
    df_right.rename(columns={'right': 'image_file'}, inplace=True)
    df_right.steering -= steer_offset

    return pd.concat([df_center, df_left, df_right])


def make_df(file_name):
    """
    Make the post-processed data set

    Make the pandas.DataFrame from a csv file and perform all pre-processing
    tasks including filtering, adding steering offset, collapsing and down
    sampling the zero-steer data.
    """
    df = pd.read_csv(file_name)
    df_filtered = filter_01(df)
    df_collapsed = collapse(df_filtered, 0.2)
    df_down_sampled = down_sample_zeros(df_collapsed, 500)
    df_down_sampled.index = range(df_down_sampled.shape[0])
    return df_down_sampled


def augment_brightness(image_in):
    """Randomly chane the brightness of an image"""
    image_out = cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image_out[:, :, 2] = image_out[:, :, 2] * random_bright
    image_out = cv2.cvtColor(image_out, cv2.COLOR_HSV2RGB)
    return image_out


def process_image(img):
    """Crop and scale the original Udacity data set image"""
    crop = img[50:-24, :, :]
    return cv2.resize(crop, (160, 43), interpolation=cv2.INTER_AREA)


def idx_init(idx, shuffle):
    """Initialize i_start and idx for a complete pass of training data"""
    if shuffle:
        np.random.shuffle(idx)
    i_start = 0
    return i_start


def image_data_generator(df, batch_size=32, shuffle=False):
    """
    Generator for to be used by keras' fit_generator method.

    Generate a batch and indefinitely loop through the training data. Real-time
    data augmentation is also handled here.
    """

    idx = df.index.copy().values
    n_images = df.shape[0]

    sample_image = cv2.imread('data/' + df.image_file.iloc[0].strip())
    image_shape = process_image(sample_image).shape

    i_start = idx_init(idx, shuffle)

    while True:
        # Calculated the ending index for this batch
        i_end = np.min([i_start + batch_size, n_images])
        # Calculate the number of samples in this batch (could be less than
        # batchsize when approaching the end of the training set)
        n = i_end-i_start
        # Initialize the batch image and label arrays
        batch_image_size = np.concatenate([[n], image_shape])
        batch_image = np.zeros(batch_image_size, dtype=np.uint8)
        batch_steer = np.zeros([i_end - i_start, 1])
        # For each sample in this batch
        for j, k in enumerate(range(i_start, i_end)):
            # Add gaussian noise to the steering values
            steer = df.loc[idx[k], 'steering'] + np.random.randn() * 0.02
            img = cv2.imread('data/' + df.loc[idx[k], 'image_file'].strip())
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = augment_brightness(img)
            # Flip a coin to decide wether to flip
            b_flip = np.random.random_integers(0, 1)
            if b_flip:
                img = cv2.flip(img, 1)
                steer = -steer

            batch_image[j] = process_image(img)
            batch_steer[j] = steer
        yield batch_image, batch_steer

        if i_end == n_images:  # End of the training set, re-initialize
            i_start = idx_init(idx, shuffle)
        else:
            i_start = i_end


def validation_set_generator(df, batch_size=32):

    idx = df.index.copy().values
    n_images = df.shape[0]

    sample_image = cv2.imread('data/' + df.image_file.iloc[0].strip())
    image_shape = process_image(sample_image).shape

    i_start = 0

    while True:
        i_end = np.min([i_start + batch_size, n_images])
        n = i_end-i_start
        batch_image_size = np.concatenate([[n], image_shape])
        batch_image = np.zeros(batch_image_size, dtype=np.uint8)
        batch_steer = np.zeros([i_end - i_start, 1])
        # For each sample in this batch
        for j, k in enumerate(range(i_start, i_end)):
            steer = df.loc[idx[k], 'steering']
            img = cv2.imread('data/' + df.loc[idx[k], 'image_file'].strip())
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_image[j] = process_image(img)
            batch_steer[j] = steer
        yield batch_image, batch_steer

        if i_end == n_images:  # End of the set, re-initialize
            i_start = 0
        else:
            i_start = i_end
