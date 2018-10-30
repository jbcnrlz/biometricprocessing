import os, random, numpy as np, math, argparse
from helper.functions import getFilesInPath
from PIL import Image as im
from keras.models import Model
from keras.utils import to_categorical
from networks.Keras.center_loss import *

def generateData(pathFiles):
    returnDataImages = []
    returnDataClass = []
    filesOnPath = getFilesInPath(pathFiles)
    for f in filesOnPath:
        if f[-3:] == 'png':
            returnDataImages.append(f)
            classNumber = f.split(os.path.sep)[-1]
            classNumber = classNumber.split('_')[0]
            returnDataClass.append(int(classNumber))

    return returnDataImages, returnDataClass

def generateImageData(paths,resize=None):
    returningPaths = []
    for p in paths:
        ni = im.open(p)
        if not resize is None:
            ni = ni.resize(resize,im.ANTIALIAS)

        ni = np.array(ni)
        if ni.shape == (100,100,4):
            returningPaths.append(np.array(ni))
        else:
            print(p)
            print('oi')
    return np.array(returningPaths)

def getBaseGallery(dataFaces,classesFaces):
    returnBaseGallery = {}
    restOfData = []
    classesRestOfData = []
    for d in range(len(classesFaces)):
        currClass = classesFaces[d]
        if not currClass in returnBaseGallery.keys():
            returnBaseGallery[currClass] = dataFaces[d]
        else:
            classesRestOfData.append(currClass)
            restOfData.append(dataFaces[d])

    return list(returnBaseGallery.values()), list(returnBaseGallery.keys()), restOfData, classesRestOfData

def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    output_filename = 'frgc_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)

def write_label_file(labels_to_class_names, dataset_dir,filename):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))

def generateDataFromArray(arrayData,Classes,batchSize,classesQtd):
    dataReturn = np.zeros((batchSize, arrayData.shape[1], arrayData.shape[2], arrayData.shape[3]))
    classReturn = np.zeros(batchSize)
    currIdx = 0
    while True:
        for i in range(arrayData.shape[0]):
            dataReturn[currIdx] = arrayData[i]
            classReturn[currIdx] = Classes[i]
            if currIdx == (dataReturn.shape[0] - 1):
                yield (dataReturn,to_categorical(classReturn - 1, num_classes=classesQtd))
                if ((arrayData.shape[0] - i) < batchSize):
                    dataReturn = np.zeros((arrayData.shape[0] - i, arrayData.shape[1], arrayData.shape[2], arrayData.shape[3]))
                    classReturn = np.zeros(arrayData.shape[0] - i)
                else:
                    dataReturn = np.zeros((batchSize, arrayData.shape[1], arrayData.shape[2], arrayData.shape[3]))
                    classReturn = np.zeros(batchSize)
                currIdx = 0
            else:
                currIdx += 1

def separateBetweenValandTrain(data,classes,percVal=0.2):
    valSize = math.ceil(data.shape[0] * percVal)
    indexesVal = random.sample([i for i in range(data.shape[0])],valSize)
    returnTrainData = []
    returnTrainClasses = []
    returnValidationData = []
    returnValidationClasses = []
    for i in range(data.shape[0]):
        if i in indexesVal:
            returnValidationClasses.append(classes[i])
            returnValidationData.append(data[i])
        else:
            returnTrainData.append(data[i])
            returnTrainClasses.append(classes[i])

    return np.array(returnTrainData), np.array(returnTrainClasses), np.array(returnValidationData), np.array(returnValidationClasses)

def prepareNetwork(network):
    model = None
    uModel = None
    if network == 'alexnet':

        x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model(numClasses=args.classNumber)

        model = Model(inputs=img_input, outputs=x)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        uModel = Model(inputs=img_input, outputs=x)
        uModel.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    elif network == 'vggcifar10':
        model = base_model(numClasses=args.classNumber)
        uModel = base_model(numClasses=args.classNumber)

    elif network == 'vgg16':
        model = base_model(foldGallery.shape[1],foldGallery.shape[2],1,args.classNumber)
        uModel = base_model(foldGallery.shape[1],foldGallery.shape[2],1,args.classNumber)
        np.rollaxis(foldGallery,3,1)

    elif network == 'vggface':
        model,img_input = base_model((100,100,4),args.classNumber)
        model = Model(inputs=img_input, outputs=model)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                    metrics=['accuracy'])

        uModel, img_input = base_model((100, 100, 4), args.classNumber)
        uModel = Model(inputs=img_input, outputs=uModel)
        uModel.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model, uModel

def separateAugmentedData(dataFiles,classesFiles):
    normalData = []
    normalDataClasses = []
    augData = []
    augDataClasses = []
    for i, d in enumerate(dataFiles):
        if 'rotate' in d:
            augData.append(d)
            augDataClasses.append(classesFiles[i])
        else:
            normalData.append(d)
            normalDataClasses.append(classesFiles[i])

    return normalData, normalDataClasses, augData, augDataClasses

def getAugData(originalFile,dataFiles,classes):
    returnData = []
    fileNameWithoutExtension = originalFile[0:-4]
    for i in range(len(dataFiles)):
        augFrom = dataFiles[i].split(os.path.sep)[-1]
        augFrom = augFrom[0:len(fileNameWithoutExtension)]
        if fileNameWithoutExtension == augFrom:
            returnData.append(dataFiles[i])

    return returnData

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-n', '--network', default=None, help='Name for the model', required=False)
    parser.add_argument('-p', '--pathBase',default='generated_images_lbp_frgc',help='Path for faces', required=False)
    parser.add_argument('-b', '--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('-c', '--classNumber', type=int, default=466, help='Quantity of classes', required=False)
    parser.add_argument('-t', '--runOnTest', type=bool, default=False, help='Run on test data', required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to be run', required=False)
    parser.add_argument('-f', '--folds', type=int, default=10, help='Fold quantity', required=False)
    parser.add_argument('-y', '--h5Path', default=None, help='Path to build h5 files', required=False)
    parser.add_argument('--onlyOnGallery', default=False, help='Utilize augmented data only on Gallery', type=bool, required=False)
    parser.add_argument('--normalization', default=None, help='Normalization function', required=False)
    args = parser.parse_args()

    if args.network == 'alexnet':
        from networks.Keras.small_AlexNet import *
    elif args.network == 'vggcifar10':
        from networks.Keras.VGGCifar10 import *
    elif args.network == 'vgg16':
        from networks.Keras.vgg16 import *
    elif args.network == 'vggface':
        from networks.Keras.otherVGGFace import *

    normFunction = args.normalization
    if (normFunction is not None):
        fName = args.normalization.split('.')[-2:]
        moduleName = args.normalization.split('.')[0:-2]
        moduleName = '.'.join(moduleName)
        normFunction = __import__(moduleName)
        for func in fName:
            normFunction = getattr(normFunction,func)

    imageData, classesData = generateData(args.pathBase)
    #gBase, cgBase, rData, crData = getBaseGallery(imageData,classesData)
    foldSize = int(len(imageData) / args.folds)
    foldResult = []
    if not args.onlyOnGallery:
        for foldNumber in range(args.folds):
            print('Fazendo fold ' + str(foldNumber))
            foldChoices = random.sample([i for i in range(len(imageData))], foldSize)
            foldProbe = []
            foldProbeClasses = []
            foldGallery = []
            foldGalleryClasses = []
            for i in range(len(imageData)):
                if i in foldChoices:
                    foldProbe.append(imageData[i])
                    foldProbeClasses.append(classesData[i])
                else:
                    foldGallery.append(imageData[i])
                    foldGalleryClasses.append(classesData[i])

            #foldGallery = generateImageData(foldGallery)
            #foldGalleryClasses = np.array(foldGalleryClasses)
            foldResult.append([foldGallery,foldGalleryClasses,foldProbe,foldProbeClasses])
    else:
        normalData, ndc, augData, adc = separateAugmentedData(imageData,classesData)
        foldSize = int(len(normalData) / args.folds)
        for foldNumber in range(args.folds):
            print('Fazendo fold '+str(foldNumber))
            foldChoices = random.sample([i for i in range(len(normalData))], foldSize)
            foldProbe = []
            foldProbeClasses = []
            foldGallery = []
            foldGalleryClasses = []

            for i in range(len(normalData)):
                if i in foldChoices:
                    foldProbe.append(normalData[i])
                    foldProbeClasses.append(ndc[i])
                else:
                    foldGallery.append(normalData[i])
                    foldGalleryClasses.append(ndc[i])

                    aDataCur = getAugData(normalData[i].split(os.path.sep)[-1], augData, adc)
                    cDataCur = [ndc[i]] * len(aDataCur)

                    foldGallery = foldGallery + aDataCur
                    foldGalleryClasses = foldGalleryClasses + cDataCur

            #foldGallery = generateImageData(foldGallery)
            #foldGalleryClasses = np.array(foldGalleryClasses)
            foldResult.append([foldGallery,foldGalleryClasses,foldProbe,foldProbeClasses])

    efr = []

    for foldData in range(len(foldResult)):
        print('Starting the fold data '+str(foldData))
        foldProbe = foldResult[foldData][2]
        foldProbeClasses = foldResult[foldData][3]
        foldGallery = generateImageData(foldResult[foldData][0])
        foldGalleryClasses = np.array(foldResult[foldData][1])
        print('Iniciando extração')
        checkpoint_path = None if args.network is None else "training/"+str(foldData)+"/face_"+args.network+"-{epoch:04d}.ckpt"

        if not os.path.exists('training'):
            os.makedirs('training')

        if not os.path.exists(os.path.join('training',str(foldData))):
            os.makedirs(os.path.join('training',str(foldData)))

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=5)

        #bd_callback = TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True, write_images=False)

        if normFunction is None:
            foldGallery = foldGallery / 255
        else:
            for i in range(len(foldGallery)):
                for j in range(foldGallery[i].shape[2]):
                    foldGallery[i,:,:,j] = np.array(normFunction(foldGallery[i,:,:,j].flatten())).reshape((100,100))

        if not args.h5Path is None:
            from helper.lmdbGeneration import h5Format

            foldProbe = generateImageData(foldProbe)

            if normFunction is None:
                foldProbe = foldProbe / 255
            else:
                for i in range(len(foldProbe)):
                    for j in range(foldProbe[i].shape[2]):
                        foldProbe[i, :, :, j] = np.array(normFunction(foldProbe[i, :, :, j].flatten())).reshape((100, 100))

            #foldProbe = foldProbe / 255
            foldProbeClasses = np.array(foldProbeClasses)

            h5Format(os.path.join(args.h5Path,'train_files.h5'),foldGallery,foldGalleryClasses)
            h5Format(os.path.join(args.h5Path, 'test_files.h5'), foldProbe, foldProbeClasses)


        #foldGallery, foldGalleryClasses, valData, valCasses = separateBetweenValandTrain(foldGallery, foldGalleryClasses)

        #valData = valData / 255
        model, uModel = prepareNetwork(args.network)

        if not args.network is None:
            model.fit_generator(
                generateDataFromArray(foldGallery,foldGalleryClasses, args.batch,args.classNumber),
                steps_per_epoch=math.ceil(foldGallery.shape[0] / args.batch),
                verbose=1,
                epochs=args.epochs,
                #validation_data=(valData,to_categorical(valCasses - 1, num_classes=args.classNumber)),
                callbacks=[cp_callback]

            )

        if args.runOnTest:
            model, uModel = prepareNetwork(args.network)
            y_binary = to_categorical(np.array(foldProbeClasses) - 1, num_classes=args.classNumber)
            foldProbe = generateImageData(foldProbe)
            if normFunction is None:
                foldProbe = foldProbe / 255
            else:
                for i in range(len(foldProbe)):
                    for j in range(foldProbe[i].shape[2]):
                        foldProbe[i, :, :, j] = np.array(normFunction(foldProbe[i, :, :, j].flatten())).reshape((100, 100))

            #foldProbe = np.array(generateImageData(foldProbe)) / 255.0
            a, accut = uModel.evaluate(foldProbe, y_binary)
            print("Untrained model, accuracy: {:5.2f}%".format(100 * accut))
            efr.append([accut])
            print('\n======================\n')
            for i in range(5, args.epochs+1,5):
                model.load_weights('training/%d/face_alexnet-%04d.ckpt' % (foldData,i))
                a, acc = model.evaluate(foldProbe, y_binary)
                print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
                efr[-1].append(acc)

    with open('results_fold.txt','w') as rf:
        for e in efr:
            rf.write(' '.join(list(map(str,e))) + '\n')

    print(efr)