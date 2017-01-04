import matplotlib.pyplot as plt
from utils import imageGenerator


def main():
    csvFile = '/Users/david/Documents/udacity/carnd.behavioral-cloning/data/driving_log.csv'
    out = imageGenerator2(csvFile, NBatchSize=256, BShuffle=True)
    out.__next__()

if __name__ == '__main__':
    main()
