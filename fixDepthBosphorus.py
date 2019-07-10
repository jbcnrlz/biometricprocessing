from helper.functions import getFilesInPath
from PIL import Image as im
import os, shutil, sys

if __name__ == '__main__':
    if sys.argv[1] == 'bosphorus':
        imageFiles = getFilesInPath('rgb_bosphorus_aligned')
        for i in imageFiles:
            currFilename = i.split(os.path.sep)[-1]
            if currFilename[0] != 'b':
                continue
            restofpath = i.split(os.path.sep)[:-1]
            className = currFilename.split('_')[0]
            className = int(className[2:])
            newFileName = str(className) + '_' + currFilename
            if i[-3:] == 'bmp':
                shutil.copy(i,os.path.sep.join(restofpath) + os.path.sep + newFileName)

            else:
                nImage = im.open(i)
                newFileName = newFileName[:-3] + 'bmp'
                nImage.save(os.path.sep.join(restofpath) + os.path.sep + newFileName)


            os.remove(i)
    elif sys.argv[1] == 'frgc_rgb':
        classes = {}
        imageFiles = getFilesInPath('rgb_aligned_frgc')
        for i in imageFiles:
            currFilename = i.split(os.path.sep)[-1]
            restofpath = i.split(os.path.sep)[:-1]
            className = currFilename.split('d')[0]
            if className in classes.keys():
                className = classes[className]
            else:
                classes[className] = len(classes)
                className = classes[className]
            newFileName = str(className) + '_' + currFilename

            if i[-3:] == 'bmp':
                shutil.copy(i,os.path.sep.join(restofpath) + os.path.sep + newFileName)

            else:
                nImage = im.open(i)
                newFileName = newFileName[:-3] + 'bmp'
                nImage.save(os.path.sep.join(restofpath) + os.path.sep + newFileName)
