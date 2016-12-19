from csv import reader
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random


# Training set statistics
csvFile = '/Users/david/Documents/udacity/carnd.behavioral-cloning/data/driving_log.csv'

f = open(csvFile)
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

NSamples = len(aSteerWheel)
print('Number of samples: {:d}'.format(NSamples))

plt.figure()
plt.hist(aSteerWheel, bins=50)
plt.xlabel('Steering Wheel Angle (rad)')

plt.figure()
plt.hist(vCar, bins=50)
plt.xlabel('Vehicle Speed (mph)')

# Random inspection
iSample = np.arange(NSamples)
np.random.shuffle(iSample)
iRandom = iSample[0:3]

plt.figure()
for i,j in enumerate(iRandom):
    plt.subplot(3,3,i*3+1)
    plt.imshow(mpimg.imread('data/'+leftImageName[j]))
    plt.subplot(3,3,i*3+2)
    plt.imshow(mpimg.imread('data/'+centerImageName[j]))
    plt.subplot(3,3,i*3+3)
    plt.imshow(mpimg.imread('data/'+rightImageName[j]))

plt.show()
