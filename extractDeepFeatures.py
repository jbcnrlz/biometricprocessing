from networks.small_AlexNet import *
from keras.models import Model
from helper.functions import  getFilesInPath
from PIL import Image as im
import os, numpy as np

def loadFolderImages(pathFiles):
    filesPath = getFilesInPath(pathFiles)
    returnFileData = []
    returnImageClasses = []
    for f in filesPath:
        fileName = f.split(os.path.sep)[-1]
        returnImageClasses.append(int(fileName.split('_')[0]))
        data = np.array(im.open(f))
        returnFileData.append(data)

    return np.array(returnFileData), returnImageClasses

if __name__ == '__main__':

    items,classes = loadFolderImages('generated_images_lfw')

    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model(numClasses=458)

    model = Model(inputs=img_input, outputs=x)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights('pretrained_models/small_alex/fold_7_face_alexnet-0100.ckpt')

    itLayerModel = Model(input=model.input,outputs=model.layers[-3].output)

    with open('database_lfw.txt','w') as dk:
        for i, data in enumerate(items):
            pred = itLayerModel.predict(np.expand_dims(data,axis=0))
            dk.write(' '.join(list(map(str,pred[0]))) + ' ' + str(classes[i]) + '\n')