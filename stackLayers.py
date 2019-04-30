from PIL import Image as im
from helper.functions import getFilesInPath
import argparse, numpy as np, os, shutil

def findIndex(search,list):
    for i, l in enumerate(list):
        if l[0:len(search)] == search:
            return i
    return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stack different files')
    parser.add_argument('--original', help='Original datasets', required=True)
    parser.add_argument('--stacked', help='Stacked file', required=True)
    parser.add_argument('--outputFileType', help='type of file to ouput', required=True)
    args = parser.parse_args()

    folders = args.original.split('__')

    files = [getFilesInPath(f,fullPath=False) for f in folders]

    if os.path.exists(args.stacked):
        shutil.rmtree(args.stacked)

    os.makedirs(args.stacked)

    fusedFiles = []
    for i in range(len(files[0])):
        currFile = np.array(im.open(os.path.join(folders[0],files[0][i])))
        if (len(currFile.shape) < 3):
            currFile = currFile.reshape((100,100,1))
        for j in range(1,len(files)):
            indexFiles = findIndex(files[0][i][:-4],files[j])
            imageOpened = np.array(im.open(os.path.join(folders[j],files[j][indexFiles])))
            if len(imageOpened.shape) < 3:
                imageOpened = imageOpened.reshape((100,100,1))
            currFile = imageOpened if currFile is None else np.concatenate((currFile,imageOpened),axis=2)

        fileName = files[0][i][:-4]
        if args.outputFileType == 'npy':
            np.save(os.path.join(args.stacked,fileName),currFile)
        else:
            imageFile = im.fromarray(currFile)
            fullFilePath = os.path.join(args.stacked,fileName+'.'+args.outputFileType)
            imageFile.save(fullFilePath)