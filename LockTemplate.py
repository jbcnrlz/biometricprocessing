from math import sqrt
from baseClasses.Template import *
import os, numpy as np, random, cv2
from helper.functions import outputObj, loadOBJ, scaleValues
from PIL import Image as im


class LockTemplate(Template):
    folderTemplate = None
    faceMarks = []
    layersChar = None
    overFlow = None
    underFlow = None

    def __init__(self,pathFile,subject,dataset=None):
        self.itemClass = subject
        self.faceMarks = []
        super().__init__(pathFile,None,True,dataset)

    def loadImageData(self):
        self.imageLoaded = True
        if (self.rawRepr[-3:]=='obj'):
            a, b, imageFace, y = loadOBJ(self.rawRepr)
            self.image = np.array(imageFace)
        elif self.rawRepr[-3:] == 'bmp':
            self.image = im.open(self.rawRepr).convert('L')
            self.layersChar = np.zeros((self.image.size[0], self.image.size[1], 4))
        else:
            imageRange = cv2.imread(self.rawRepr, -1)
            self.image = np.array([[i,j,imageRange[i,j]] for i in range(imageRange.shape[0]) for j in range(imageRange.shape[1])])
            #self.image = imageRange

    def loadImage(self):
        if self.rawRepr[-3:] == 'bmp':
            imageFace = im.open(self.rawRepr).convert('L')
            self.layersChar = np.zeros((imageFace.size[0], imageFace.size[1], 4))
        elif self.rawRepr[-3:] == 'png':
            imageRange = cv2.imread(self.rawRepr,-1)
            imageFace = [[i,j,imageRange[i,j]] for i in range(imageRange.shape[0]) for j in range(imageRange.shape[1]) if imageRange[i,j]]
        self.image = imageFace

    def loadSymFilledImage(self):
        if self.lazyLoading:
            self.rawRepr = self.rawRepr[0:-4] + '_symmetricfilled.obj'
        else:
            a, b, imageFace, y = loadOBJ(self.rawRepr[0:-4] + '_symmetricfilled.obj')
            self.image = imageFace

    def saveNewDepth(self,fileExtension=''):
        sImage = im.fromarray(self.image).convert('L')
        sImage.save(self.rawRepr[0:-4]+fileExtension+'_newdepth.bmp')


    def save(self, saveOnPath=False):
        if (not saveOnPath):
            if (not os.path.exists('temporaryTemplate')):
                os.makedirs('temporaryTemplate')

            outputObj(self.image, os.path.join('temporaryTemplate', str(
                self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.obj'))
            self.outputMarks()
        else:
            self.image.save(self.rawRepr[0:-4] + '_newdepth.bmp')

    def loadMarks(self):
        filesPath = self.rawRepr.split(os.path.sep)
        fileName = filesPath[-1].split('.')
        fileName = fileName[0].split('_')
        fileName = '_'.join(fileName[:-1])+'.txt'
        filesPath = os.path.sep.join(filesPath[:-1]) + os.path.sep + fileName
        with open(filesPath, 'r') as keypoints:
            for p in keypoints:
                pointsFace = p.strip().split(' ')
                self.faceMarks.append([int(pointsFace[1][pointsFace[1].index(':')+1:]),int(pointsFace[0][pointsFace[0].index(':')+1:])])

    def loadNewDepthImage(self):
        self.image = im.open(self.rawRepr[0:-4] + '_newdepth.bmp')
        self.loadMarks('newdepth')

    def saveImageTraining(self, avgImageSave=True, pathImage='generated_images_lbp'):
        # imageSaveDLP = np.array(self.layersChar)
        if (avgImageSave):
            avImage = np.zeros((self.layersChar.shape[0], self.layersChar.shape[1]))
            for i in range(self.layersChar.shape[0]):
                for j in range(self.layersChar.shape[1]):
                    avImage[i, j] = self.layersChar[i, j, 0] + self.layersChar[i, j, 1] + self.layersChar[i, j, 2] + \
                                    self.layersChar[i, j, 3]
                    avImage[i, j] = avImage[i, j] / 4
            avImage = im.fromarray(np.uint8(avImage))
            # avImage.save(pathImage+'/averageImage/'+str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath +'.jpg')

        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = im.fromarray(np.uint8(self.layersChar))
        pathNImage = pathImage + '/' + str(self.itemClass) + '_' + fullPath + '.png'
        imageSaveDLP.save(pathNImage)

    def saveHistogramImage(self, imageSave=None, folder='generated_images_wld'):
        if (imageSave is None):
            imageSave = self.features

        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = im.fromarray(imageSave)
        pathNImage = folder + '/' + str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath + '.jpg'
        while (os.path.exists(pathNImage)):
            idxRandomIm = random.randint(1, 255)
            pathNImage = folder + '/' + str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath + '_' + str(
                idxRandomIm) + '.png'

        imageSaveDLP.convert('RGB').save(pathNImage)

    def saveMasks(self, folder, filetype):
        if self.overFlow is None or self.underFlow is None:
            return None
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = None
        if not os.path.exists(folder):
            os.makedirs(folder)
        pathNImage = folder + '/' + str(
            self.itemClass) + '_' + self.folderTemplate + '_' + fullPath + '_' + filetype + '.bmp'
        if filetype == 'overflow':
            self.overFlow = scaleValues(0, 255, self.overFlow)
            imageSaveDLP = im.fromarray(self.overFlow)
        else:
            self.underFlow = scaleValues(0, 255, self.underFlow)
            imageSaveDLP = im.fromarray(self.underFlow)

        imageSaveDLP.convert('RGB').save(pathNImage)

    def isFileExists(self, pathImage, filetype='png'):
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        pathNImage = pathImage + '/' + str(self.itemClass) + '_' + fullPath + '.' + filetype
        return os.path.exists(pathNImage)

    def existsPreProcessingFile(self):
        return os.path.exists(self.rawRepr[0:-4] + '_newdepth.bmp')