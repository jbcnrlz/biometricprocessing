from helper.functions import getFilesInPath
import argparse, os, shutil, numpy as np
from PIL import Image as im

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mix channels')
    parser.add_argument('--pathBase', help='Base path', required=True)
    parser.add_argument('--outputPathBase', help='Output path', required=True)
    parser.add_argument('--channels', type=int, help='Number of channels', required=False, default=4)
    parser.add_argument('--swaped', help='swaped_list', required=False, default='2_3_0_1')
    args = parser.parse_args()

    channelsNumber = list(range(args.channels))
    newChannels = list(map(int,args.swaped.split('_')))
    files = getFilesInPath(args.pathBase)

    if os.path.exists(args.outputPathBase):
        shutil.rmtree(args.outputPathBase)

    os.makedirs(args.outputPathBase)

    for f in files:
        imageFile = im.open(f)
        fileName = f.split(os.path.sep)[-1]
        originalImage = np.array(imageFile)
        newSwapedImage = np.zeros(originalImage.shape)
        for i in channelsNumber:
            newSwapedImage[:,:,newChannels[i]] = originalImage[:,:,i]

        newImage = im.fromarray(newSwapedImage.astype(np.uint8))
        newImage.save(os.path.join(args.outputPathBase,fileName))