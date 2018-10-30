from networks.Keras.small_AlexNet import *
from keras.models import Model
from helper.functions import  getFilesInPath
from PIL import Image as im
import os, numpy as np, argparse, random, shutil, re

def loadFolderImages(pathFiles,regularExpression=None):
    filesPath = getFilesInPath(pathFiles)
    newFilePath = []
    returnFileData = []
    returnImageClasses = []
    for f in filesPath:
        fileName = f.split(os.path.sep)[-1]
        if regularExpression is None or re.match(regularExpression,fileName):
            newFilePath.append(fileName)
            returnImageClasses.append(int(fileName.split('_')[0]))
            data = np.array(im.open(f))
            returnFileData.append(data)

    return np.array(returnFileData), returnImageClasses, newFilePath

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Deep Features utilizing GioGio Network')
    parser.add_argument('-p','--pathdatabase',help='Path for the database',required=True)
    parser.add_argument('-w', '--weights', help='File for weights', required=True)
    parser.add_argument('-o', '--output', help='Output for features', default='out.txt')
    parser.add_argument('-g', '--gallery', help='Gallery', default='gallery.txt')
    parser.add_argument('-r', '--probe', help='Probe', default='probe.txt')
    parser.add_argument('-f', '--folds', help='Quantity of folds', default=None, type=int)
    parser.add_argument('--exp', help='Regular expression', default=None)
    args = parser.parse_args()

    items,classes, files = loadFolderImages(args.pathdatabase,args.exp)

    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model(numClasses=458)

    model = Model(inputs=img_input, outputs=x)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights(args.weights)

    itLayerModel = Model(input=model.input,outputs=model.layers[-3].output)

    with open(args.output,'w') as dk:
        for i, data in enumerate(items):
            print(files[i])
            data = data / 255.0
            pred = itLayerModel.predict(np.expand_dims(data,axis=0))
            dk.write(' '.join(list(map(str,pred[0]))) + ' ' + str(classes[i]) + '\n')

    if args.folds is not None:

        if not os.path.exists('cross_fold'):
            os.makedirs('cross_fold')

        if os.path.exists(os.path.join('cross_fold', 'GioGio')):
            shutil.rmtree(os.path.join('cross_fold', 'GioGio'))

        os.makedirs(os.path.join('cross_fold', 'GioGio'))

        alreadyWent = []
        sizeProbe = int(len(items) / args.folds)

        probeTextFile = args.probe.split('.')
        probeTextFile = probeTextFile[0] + '%02d.' + probeTextFile[1]

        galleryTextFile = args.gallery.split('.')
        galleryTextFile = galleryTextFile[0] + '%02d.' + galleryTextFile[1]

        for fn in range(args.folds):

            os.makedirs(os.path.join('cross_fold', 'GioGio',str(fn)))

            foldChoices = random.sample([i for i in range(len(items))], sizeProbe)
            probe = []
            gallery = []
            for fIdx in range(len(foldChoices)):
                while(foldChoices[fIdx] in alreadyWent):
                    newSample = random.randint(0,len(files)-1)
                    if newSample not in foldChoices:
                        foldChoices[fIdx] = newSample

            for fileIdx, file in enumerate(files):
                if fileIdx in foldChoices:
                    probe.append(file+ ' ' +str(fileIdx))
                else:
                    gallery.append(file+ ' ' +str(fileIdx))

            alreadyWent = alreadyWent + foldChoices

            with open(os.path.join('cross_fold', 'GioGio',str(fn),galleryTextFile) % (fn),'w') as foldFile:
                for p in gallery:
                    foldFile.write(p + '\n')


            with open(os.path.join('cross_fold', 'GioGio',str(fn),probeTextFile) % (fn),'w') as foldFile:
                for p in probe:
                    foldFile.write(p + '\n')

        shutil.copy(args.output,os.path.join('cross_fold','GioGio',args.output))