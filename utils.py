from csv import reader
import numpy as np
import math, cv2

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
        crop = img[:,int(dx/2):-int(dx/2),:]
    elif input_aspect_ratio < ASPECT_RATIO:
        # print('Aspect ratio is too small - reduce height')
        dy = int(height - width / ASPECT_RATIO)
        # img_overlay = img.copy()
        # img_overlay = cv2.rectangle(img_overlay, (0, dy), (width, height), (0, 255, 0), 3)
        crop = img[dy:,:,:]

    # Using INTER_AREA assuming shrinking
    return cv2.resize(crop, (200, 80), interpolation = cv2.INTER_AREA)


def imageGenerator(file_name, NBatchSize=1, BShuffle=False):

    delta_steering = 0.15

    f = open(file_name)
    csvReader = reader(f)

    image_name = []
    aSteerWheel = []
    header = csvReader.__next__()

    # TODO is there a way to sniff the csv and pre allocate?
    for i,row in enumerate(csvReader):
        image_name.append(row[0].strip()) # center
        image_name.append(row[1].strip()) # left
        image_name.append(row[2].strip()) # right
        a = np.single(row[3])
        aSteerWheel.append(a) # center
        aSteerWheel.append(a + delta_steering) # left
        aSteerWheel.append(a - delta_steering) # right

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

    while 1:
        for i in range(NBatches):
            # Calculate the batch indices
            iStart = i*NBatchSize
            iEnd = np.min([iStart+NBatchSize, NImages])
            N = iEnd-iStart

            batch_image_name = image_name[iStart:iEnd]
            batch_image = np.zeros(np.concatenate([[N], imageShape]), dtype=np.uint8)

            for j, f in enumerate(batch_image_name):
                img = cv2.imread('data/'+f)
                batch_image[j] = process_image(img)

            batch_steering = np.array(aSteerWheel[iStart:iEnd])

            yield batch_image, batch_steering
