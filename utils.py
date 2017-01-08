from csv import reader
import numpy as np
import math, cv2

def process_image(img):
    """ Crop from the top if croping vertically or crop equally from the sides
    if croping horizontally
    """
    # height, width = img.shape[:2]
    # Resize to 65% of the original size
    # res = cv2.resize(img, (int(width*0.65), int(height*0.65)), interpolation = cv2.INTER_AREA)
    # Crop the top 30% off
    # crop = res[int(res.shape[0]*.3):,:,:]
    # return crop
    aspect_ratio = 2.5
    height, width = img.shape[:2]

    if width/height < aspect_ratio:
        dy = int(height - width / aspect_ratio)
        crop = img[dy:,:,:]
        # height, width = crop.shape[:2]
        # print(width/height)
    elif width/height > aspect_ratio:
        dx = width - height * aspect_ratio
        crop = img[:,int(dx/2):-int(dx/2),:]
        # height, width = crop.shape[:2]
        # print(width/height)

    # Using INTER_AREA assuming shrinking
    res = cv2.resize(crop, (200, 80), interpolation = cv2.INTER_AREA)
    return res

def imageGenerator(file_name, NBatchSize=1, BShuffle=False):

    f = open(file_name)
    csvReader = reader(f)

    leftImageName = []
    centerImageName = []
    rightImageName = []
    aSteerWheel = []
    vCar = []
    header = csvReader.__next__()

    # TODO is there a way to sniff the csv and pre allocate?
    for i,row in enumerate(csvReader):
        leftImageName.append(row[1].strip())
        centerImageName.append(row[0].strip())
        rightImageName.append(row[2].strip())
        aSteerWheel.append(np.single(row[3]))
        vCar.append(np.single(row[6]))

    NImages = len(aSteerWheel)
    NBatches = math.ceil(NImages/NBatchSize)

    if BShuffle:
        i = np.arange(NImages, dtype=np.uint8)
        np.random.shuffle(i)
        leftImageName = [leftImageName[idx] for idx in i]
        centerImageName = [centerImageName[idx] for idx in i]
        rightImageName = [rightImageName[idx] for idx in i]
        aSteerWheel = [aSteerWheel[idx] for idx in i]
        vCar = [vCar[idx] for idx in i]

    tmp = cv2.imread('data/'+centerImageName[0])
    tmp = process_image(tmp)
    imageShape = tmp.shape

    while 1:
        for i in range(NBatches):
            # Calculate the batch indices
            iStart = i*NBatchSize
            iEnd = np.min([iStart+NBatchSize, NImages])
            N = iEnd-iStart

            batchLeftImageName = leftImageName[iStart:iEnd]
            batchCenterImageName = centerImageName[iStart:iEnd]
            batchRightImageName = rightImageName[iStart:iEnd]

            # lImage = np.zeros(np.concatenate([[N], imageShape]), dtype=np.uint8)
            cImage = np.zeros(np.concatenate([[N], imageShape]), dtype=np.uint8)
            # rImage = np.zeros(np.concatenate([[N], imageShape]), dtype=np.uint8)

            for j,(lImageName, cImageName, rImageName) in enumerate(zip(batchLeftImageName, batchCenterImageName, batchRightImageName)):
                # lImage[j] = mpimg.imread('data/'+lImageName)
                img = cv2.imread('data/'+cImageName)
                cImage[j] = process_image(img)
                # rImage[j] = mpimg.imread('data/'+rImageName)

            # yield (lImage, cImage, rImage, aSteerWheel[iStart:iEnd], vCar[iStart:iEnd])
            yield (cImage, np.array(aSteerWheel[iStart:iEnd]))
