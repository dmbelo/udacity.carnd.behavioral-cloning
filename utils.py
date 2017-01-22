import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot(df):
    ax1 = plt.subplot(411)
    plt.plot(df.steering, 'o')

    plt.subplot(412, sharex=ax1)
    plt.plot(df.throttle, 'o')

    plt.subplot(413, sharex=ax1)
    plt.plot(df.brake, 'o')

    plt.subplot(414, sharex=ax1)
    plt.plot(df.speed, 'o')

    plt.show(block=False)


def play(df):
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
    b_zero_steer = df.steering == 0
    idx = b_zero_steer.index[b_zero_steer].values
    np.random.shuffle(idx)
    idx_remove_zeros = idx[zeros:]
    return df.drop(idx_remove_zeros, axis=0)


def filter_01(df):
    b_ends = (df.index < 80) | (df.index > 7790)
    b_middle = (df.index > 3400) & (df.index < 4600)
    return df.loc[(~b_ends) & (~b_middle), :].copy()


def collapse(df, steer_offset):
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
    df = pd.read_csv(file_name)
    df_filtered = filter_01(df)
    df_down_sampled = down_sample_zeros(df_filtered, 500)
    df_collapsed = collapse(df_down_sampled, 0.25)
    df_collapsed.index = range(df_collapsed.shape[0])
    return df_collapsed


def process_image(img):
    ASPECT_RATIO = 2.5
    height, width = img.shape[:2]
    input_aspect_ratio = width/height
    if input_aspect_ratio > ASPECT_RATIO:
        dx = width - height * ASPECT_RATIO
        crop = img[:, int(dx/2):-int(dx/2), :]
    elif input_aspect_ratio < ASPECT_RATIO:
        dy = int(height - width / ASPECT_RATIO)
        crop = img[dy:, :, :]

    # Using INTER_AREA assuming shrinking
    return cv2.resize(crop, (200, 80), interpolation=cv2.INTER_AREA)


def epoch_init(idx, shuffle):
    if shuffle:
        np.random.shuffle(idx)
    return 0


def image_data_generator(df, batch_size=32, shuffle=False):

    idx = df.index.copy().values
    n_images = df.shape[0]

    sample_image = cv2.imread('data/' + df.image_file.iloc[0].strip())
    image_shape = process_image(sample_image).shape

    i_start = epoch_init(idx, shuffle)

    while True:
        i_end = np.min([i_start + batch_size, n_images])
        n = i_end-i_start
        batch_image_size = np.concatenate([[n], image_shape])
        batch_image = np.zeros(batch_image_size, dtype=np.uint8)
        batch_steer = np.zeros([i_end - i_start, 1])

        for j, k in enumerate(range(i_start, i_end)):
            steer = df.loc[idx[k], 'steering'] + np.random.randn() * 0.025
            img = cv2.imread('data/' + df.loc[idx[k], 'image_file'].strip())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            b_flip = np.random.random_integers(0, 1)
            if b_flip:
                img = cv2.flip(img, 1)
                steer = -steer

            batch_image[j] = process_image(img)
            batch_steer[j] = steer
        yield batch_image, batch_steer

        if i_end == n_images:
            i_start = epoch_init(idx, shuffle)
        else:
            i_start = i_end
