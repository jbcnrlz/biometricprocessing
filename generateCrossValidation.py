import os, shutil, random, numpy as np, tensorflow as tf, math
from helper.functions import getFilesInPath
from PIL import Image as im
from networks.alexnet import *
from AlexNet import *
from keras.models import Model
from keras.utils import to_categorical

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

def generateImageData(paths):
    returningPaths = []
    for p in paths:
        ni = im.open(p)
        #ni = ni.resize((224,224),im.ANTIALIAS)
        returningPaths.append(np.array(ni))
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

def generateDataFromArray(arrayData,Classes,batchSize):
    dataReturn = np.zeros((batchSize, arrayData.shape[1], arrayData.shape[2], arrayData.shape[3]))
    classReturn = np.zeros(batchSize)
    currIdx = 0
    while True:
        for i in range(arrayData.shape[0]):
            dataReturn[currIdx] = arrayData[i]
            classReturn[currIdx] = Classes[i]
            if currIdx == batchSize - 1:
                yield (dataReturn,to_categorical(classReturn - 1, num_classes=466))
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
            returnTrainData.append(classes[i])
            returnTrainClasses.append(classes[i])

    return np.array(returnTrainData), np.array(returnTrainClasses), np.array(returnValidationData), np.array(returnValidationClasses)

if __name__ == '__main__':
    imageData, classesData = generateData('generated_images_lbp_frgc')
    gBase, cgBase, rData, crData = getBaseGallery(imageData,classesData)
    foldSize = int(len(rData) / 10)
    foldResult = []

    foldChoices = random.sample([i for i in range(len(rData))], foldSize)
    foldProbe = []
    foldProbeClasses = []
    foldGallery = []
    foldGalleryClasses = []
    for i in range(len(rData)):
        if i in foldChoices:
            foldProbe.append(rData[i])
            foldProbeClasses.append(crData[i])
        else:
            foldGallery.append(rData[i])
            foldGalleryClasses.append(crData[i])

    foldGallery = generateImageData(gBase + foldGallery)
    foldGalleryClasses = np.array(cgBase + foldGalleryClasses)

    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    model = Model(inputs=img_input,outputs=x)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                metrics=['accuracy'])

    checkpoint_path = "training/face_alexnet-{epoch:04d}.ckpt"

    if os.path.exists('training'):
        shutil.rmtree('training')

    os.makedirs('training')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=1)

    foldGallery, foldGalleryClasses, valData, valCasses = separateBetweenValandTrain(foldGallery,foldGalleryClasses)

    model.fit_generator(
        generateDataFromArray(foldGallery,foldGalleryClasses,500),
        steps_per_epoch=foldGallery.shape[0],
        verbose=1,
        epochs=10,
        validation_data=(valData,to_categorical(valCasses - 1, num_classes=466)),
        callbacks=[cp_callback]

    )

    #y_binary = to_categorical(foldGalleryClasses - 1, num_classes=466)
    #model.fit(foldGallery, y_binary, epochs=10, batch_size=10, callbacks=[cp_callback])
    '''
    dataForTraining,classesData = generateBatchForTraining(foldGallery,foldGalleryClasses - 1,466)

    for d in range(len(dataForTraining)):
        y_binary = to_categorical(np.array(classesData[d]) - 1,num_classes=466)
        td = np.array(dataForTraining[d])
        model.fit(td,y_binary,epochs=10,batch_size=32,callbacks = [cp_callback])
    '''
    '''

    y_binary = to_categorical(np.array(foldProbeClasses) -1,num_classes=224)
    foldProbe = np.array(generateImageData(foldProbe))
    a, acc = model.evaluate(foldProbe,y_binary)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    print('\n======================\n')
    for i in range(1,6):
        model.load_weights('training/face_alexnet-000'+str(i)+'.ckpt')
        a, acc = model.evaluate(np.array(foldProbe),y_binary)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    '''