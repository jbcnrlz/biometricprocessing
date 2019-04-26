from helper.functions import getFilesInPath
import numpy as np, argparse, os, shutil
from PIL import Image as im
from random import randint, choice

def rotateImage(image,angle):
    return image.rotate(angle)

def randomScale(image):
    w, h = image.size
    randomWidth = randint(int(w/2),w*2)
    wpercent = (randomWidth/float(w))
    hsize = int((float(h)*float(wpercent)))
    nImg = image.resize((randomWidth,hsize), im.ANTIALIAS)

    imageData = np.array(nImg)
    nImageResized = np.zeros((w, h, imageData.shape[2]),dtype=np.uint8)
    for i in range(imageData.shape[0]):
        if i >= w:
            break
        for j in range(imageData.shape[1]):
            if j >= h:
                break

            nImageResized[i,j] = imageData[i,j]

    return im.fromarray(nImageResized)

def removeLowerBonds(image):
    w, h = image.size
    imageData = np.array(image)
    nImageResized = np.zeros((w, h, imageData.shape[2]),dtype=np.uint8)
    for i in range(int(h/2)):
        for j in range(imageData.shape[1]):
            nImageResized[i,j] = imageData[i,j]
    return im.fromarray(nImageResized)

def randomTranslation(image):
    w, h = image.size
    limitx, limity = int(w/2), int(h/2)
    tx = randint(10, limitx) * choice([-1,1])
    ty = randint(10, limity) * choice([-1, 1])
    nImg = image.transform(image.size, im.AFFINE, (1, 0, tx, 0, 1, ty))
    return nImg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Roughly do data augmentation')
    parser.add_argument('--loadFromFolder', help='Load folds from folders', required=True)
    parser.add_argument('--augFolder', help='Fold to save augData', required=False)
    parser.add_argument('--angles', default=None, help='2D Rotations', required=False)
    parser.add_argument('--randomScale', default=None, help='Quantity of random scale', required=False)
    parser.add_argument('--removeUnder', default=None, help='Remove the under part of the images', required=False)
    parser.add_argument('--randomTranslation', default=False, help='Quantity of random translations', required=False)
    parser.add_argument('--mixDeforms', default=False, help='Numberofdeform_numberofmix', required=False)
    args = parser.parse_args()

    if args.augFolder == args.loadFromFolder:
        raise Exception('Cannot be the same')

    if os.path.exists(args.augFolder):
        shutil.rmtree(args.augFolder)

    os.makedirs(args.augFolder)

    files = getFilesInPath(args.loadFromFolder)
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        print('Doing ' + fileName + ' file')
        fileName = fileName.split('.')
        shutil.copy(f,os.path.join(args.augFolder,'.'.join(fileName)))
        currImage = im.open(f)

        if args.angles is not None:
            anglesRotate = args.angles.split('_')
            for a in anglesRotate:
                nImage = rotateImage(currImage,int(a))
                prefix = '_rotate_%03d_roughaug' % (int(a))
                nImage.save(os.path.join(args.augFolder,'.'.join([fileName[0]+prefix,fileName[1]])))

        if args.randomScale is not None:
            for i in range(int(args.randomScale)):
                nImage = randomScale(currImage)
                prefix = '_scale_%02d_roughaug' % (i)
                nImage.save(os.path.join(args.augFolder,'.'.join([fileName[0]+prefix,fileName[1]])))

        if args.removeUnder is not None:
            nImage = removeLowerBonds(currImage)
            prefix = '_under_removed_roughaug'
            nImage.save(os.path.join(args.augFolder, '.'.join([fileName[0] + prefix, fileName[1]])))

        if args.randomTranslation is not None:
            for i in range(int(args.randomTranslation)):
                nImage = randomTranslation(currImage)
                prefix = '_translate_%02d_roughaug' % (i)
                nImage.save(os.path.join(args.augFolder,'.'.join([fileName[0]+prefix,fileName[1]])))

        if args.mixDeforms is not None:
            qtEfects, qtImages = args.mixDeforms.split('_')
            anglesRotate = [45,90,135,180] if args.angles is None else [int(angv) for angv in args.angles.split('_')]
            nImage = currImage
            for j in range(int(qtImages)):
                for i in range(int(qtEfects)):
                    chosedDeform = choice([1,2,3,4])
                    if chosedDeform == 1:
                        nImage = rotateImage(nImage,choice(anglesRotate))
                    elif chosedDeform == 2:
                        nImage = randomScale(nImage)
                    elif chosedDeform == 3:
                        nImage = removeLowerBonds(nImage)
                    elif chosedDeform == 4:
                        nImage = randomTranslation(nImage)

                prefix = '_randomize_%03d_roughaug' % (j)
                nImage.save(os.path.join(args.augFolder,'.'.join([fileName[0]+prefix,fileName[1]])))
