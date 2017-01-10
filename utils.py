import numpy as np
import math
import cv2


def process_image(img):
    ASPECT_RATIO = 2.5

    height, width = img.shape[:2]
    # print('height:', height)
    # print('width:', width)

    input_aspect_ratio = width/height
    # print('input aspect ratio:', input_aspect_ratio)

    if input_aspect_ratio > ASPECT_RATIO:
        # print('Aspect ratio too great - reduce width')
        dx = width - height * ASPECT_RATIO
        # img_overlay = img.copy()
        # img_overlay = cv2.rectangle(img_overlay, (int(dx/2), 0), (width-int(dx/2), height), (0, 255, 0), 3)
        crop = img[:, int(dx/2):-int(dx/2), :]
    elif input_aspect_ratio < ASPECT_RATIO:
        # print('Aspect ratio is too small - reduce height')
        dy = int(height - width / ASPECT_RATIO)
        # img_overlay = img.copy()
        # img_overlay = cv2.rectangle(img_overlay, (0, dy), (width, height), (0, 255, 0), 3)
        crop = img[dy:, :, :]

    # Using INTER_AREA assuming shrinking
    return cv2.resize(crop, (200, 80), interpolation=cv2.INTER_AREA)


def trim_zero_steer(steer, n_trim):

    N = len(steer)
    idx = np.arange(N)
    idx_zero = idx[steer == 0]
    np.random.shuffle(idx_zero)
    idx_non_zero = idx[steer != 0]
    idx_trimmed = np.sort(np.concatenate([idx_non_zero, idx_zero[:n_trim]]))

    return idx_trimmed


def parse_csv(file_name, delta_steering=0.08, n_trim=None):
    raw = np.genfromtxt(file_name,
                        skip_header=1,
                        delimiter=',',
                        autostrip=True,
                        dtype=("|S50", "|S50", "|S50",
                               np.single, np.single, np.single, np.single))

    img_file, steer = zip(*[[[x[0].decode('utf-8'),
                              x[1].decode('utf-8'),
                              x[2].decode('utf8')], x[3]] for x in raw])

    img_file = np.array(img_file)
    img_file_c = img_file[:, 0]
    img_file_l = img_file[:, 1]
    img_file_r = img_file[:, 2]
    steer_c = np.array(steer)
    steer_l = steer_c + np.float32(delta_steering)
    steer_r = steer_c - np.float32(delta_steering)

    if n_trim:
        idx = trim_zero_steer(np.array(steer_c), n_trim)
        img_file_c = img_file_c[idx]
        img_file_l = img_file_l[idx]
        img_file_r = img_file_r[idx]
        steer_c = steer_c[idx]
        steer_l = steer_l[idx]
        steer_r = steer_r[idx]

    img_file_flat = np.concatenate([img_file_c, img_file_l, img_file_r])
    steer_flat = np.concatenate([steer_c, steer_l, steer_r])

    return img_file_flat, steer_flat


def imageGenerator(image_name, aSteerWheel, NBatchSize=1, BShuffle=False):

    while True:
        NImages = len(aSteerWheel)
        NBatches = math.ceil(NImages/NBatchSize)
        if BShuffle:
            i = np.arange(NImages, dtype=np.uint8)
            np.random.shuffle(i)
            image_name = [image_name[idx] for idx in i]
            aSteerWheel = [aSteerWheel[idx] for idx in i]

        tmp = cv2.imread('data/'+image_name[0])
        tmp = process_image(tmp)
        imageShape = tmp.shape

        for i in range(NBatches):
            # Calculate the batch indices
            iStart = i*NBatchSize
            iEnd = np.min([iStart+NBatchSize, NImages])
            N = iEnd-iStart

            batch_steering = np.array(aSteerWheel[iStart:iEnd])

            batch_image_name = image_name[iStart:iEnd]
            batch_image = np.zeros(np.concatenate([[N], imageShape]),
                                   dtype=np.uint8)

            for j, f in enumerate(batch_image_name):
                img = cv2.imread('data/'+f)
                batch_image[j] = process_image(img)

            yield batch_image, batch_steering
