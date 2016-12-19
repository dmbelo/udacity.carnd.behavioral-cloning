from csv import reader
import matplotlib.pyplot as plt
import numpy as np


# Training set statistics
csv_file = '/Users/david/Documents/udacity/carnd.behavioral-cloning/data/driving_log.csv'

f = open(csv_file)
csv_reader = reader(f)

aSteerWheel = []
vCar = []
header = csv_reader.__next__()

# TODO is there a way to sniff the csv and pre allocate?
for i,row in enumerate(csv_reader):
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
plt.show()
