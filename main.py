from utils import imageGenerator


def main():
    csv_file = 'data/driving_log.csv'
    out = imageGenerator(csv_file, NBatchSize=256, BShuffle=True)
    out.__next__()

if __name__ == '__main__':
    main()
