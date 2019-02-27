from PIL import Image as im
from helper.functions import getFilesInPath
import numpy as np, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate average data for training')
    parser.add_argument('--pathFiles', default=None, help='Load images from folders', required=False)
    args = parser.parse_args()

    files = getFilesInPath(args.pathFiles)
    summedImages = [[],[],[]]
    for f in files:
        currImage = np.array(im.open(f)) / 255
        summedImages[0].append(currImage[:,:,0])
        summedImages[1].append(currImage[:, :, 1])
        summedImages[2].append(currImage[:, :, 2])

    summedImages = np.array(summedImages)
    averageValue = []
    stdValue = []
    for i in range(summedImages.shape[0]):
        averageValue.append(np.average(summedImages[i]))
        stdValue.append(np.std(summedImages[i]))

    print(averageValue)
    print(stdValue)