import numpy as np, math, operator, os, logging, scipy.ndimage as ndimage, matlab.engine, pywt
from baseClasses.BiometricProcessing import *
from helper.functions import generateHistogram, loadOBJ, generateHistogramUniform, bilinear_interpolation, mergeArraysDiff
from sklearn import svm
from helper.functions import minmax
from PIL import Image as im
import pywt
import time
from scipy.spatial.distance import *
from scipy.special import expit
from mahotas.interpolate import shift
from LFWTemplate import *

class WaveletHistogram(BiometricProcessing):

    def __init__(self,windowsize,binsize,database,width=100,height=100):
        self.width = width
        self.height = height
        self.windowSize = windowsize
        self.binsize = binsize
        self.databases = database

    def getLimits(self,fullFace):
        smallerValues = [float('inf'),float('inf')]
        biggerValues = [-float('inf'),-float('inf')]
        for f in fullFace:
            if (f[0] < smallerValues[0]):
                smallerValues[0] = f[0]

            if (f[1] < smallerValues[1]):
                smallerValues[1] = f[1]

            if (f[0] > biggerValues[0]):
                biggerValues[0] = f[0]

            if (f[1] > biggerValues[1]):
                biggerValues[1] = f[1]

        return smallerValues + biggerValues

    def generateCode(self,image):
        try:
            gradients = np.gradient(image)
            hcA, hcD = pywt.dwt(gradients[0],'db1')
            vcA, vcD = pywt.dwt(gradients[1],'db1')
            return hcA, hcD, vcA, vcD
        except:
            return None

    def setupTemplate(self,template):
        return template

    def cleanupTemplate(self,template):
        template.layersChar = np.zeros((len(template.image),len(template.image[0]),4))
        return template

    def generateSteps(self,limits,size=(100,100)):
        idxSep = []
        idySep = []
        windowXSize = 0
        if (limits[0] < 0):
            windowXSize = (limits[2] - limits[0]) / size[0]
        else:
            windowXSize = (limits[2] + limits[0]) / size[0]

        windowYSize = 0
        if (limits[1] < 0):
            windowYSize = (limits[3] - limits[1]) / size[1]
        else:
            windowYSize = (limits[3] + limits[1]) / size[1]

        currentY = limits[1]
        while(currentY <= limits[3]):
            idySep.append(currentY)
            currentY += windowYSize

        currentX = limits[0]
        while(currentX <= limits[2]):
            idxSep.append(currentX)
            currentX += windowXSize

        return idxSep, idySep

    def generateWindows(self,fullFace,limits,size=(100,100)):
        idxSep = []
        idySep = []
        windowXSize = 0
        if (limits[0] < 0):
            windowXSize = (limits[2] - limits[0]) / size[0]
        else:
            windowXSize = (limits[2] + limits[0]) / size[0]

        windowYSize = 0
        if (limits[1] < 0):
            windowYSize = (limits[3] - limits[1]) / size[1]
        else:
            windowYSize = (limits[3] + limits[1]) / size[1]

        currentY = limits[1]
        region = []
        while(currentY <= limits[3]):
            idySep.append(currentY)
            currentX = limits[0]
            while(currentX <= limits[2]):
                if not currentX in idxSep:
                    idxSep.append(currentX)
                region.append([])
                for f in fullFace:
                    if (f[0] >= currentX) and (f[0] < currentX + windowXSize) and (f[1] >= currentY) and (f[1] < currentY + windowYSize):
                        region[len(region) - 1].append(f)

                currentX += windowXSize
            currentY += windowYSize

        return region, idxSep, idySep

    def generatePixelValue(self,region,typeGen='Average'):
        if (type(region) is not np.ndarray):
            region = np.array(region)
        if typeGen == 'Average':
            return np.sum(region[0:,2]) / region.shape[0]

    def arrangeImage(self,image):
        newImg = np.full((self.height,self.width),np.inf)
        for i in image:
            for j in image[i]:
                newImg[i][j] = self.generatePixelValue(image[i][j])

        return newImg

    def normalizeImage(self,image,minv,maxv):
        if (type(image) is dict):
            image = self.arrangeImage(image)
        valuesAnalise = np.array([i for i in image.flatten() if i != np.inf])
        m = np.min(valuesAnalise)
        rangeValue = np.max(valuesAnalise) - m
        image = (image - m) / rangeValue

        range2 = maxv - minv
        image = (image*range2) + minv
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j] == np.inf:
                    image[i,j] = 0
                else:
                    image[i,j] = int(image[i,j])
        return image

    def fixSizeMatrix(self,original,fixMat,axis=1):
        shapeRigh = list(original.shape)
        shapeRigh[axis] = fixMat.shape[axis]

        righMatrix = np.zeros(shapeRigh)
        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                righMatrix[i,j] = original[i,j]

        return righMatrix

    def featureExtraction(self,excludeFile=[]):
        print("Iniciando feature extraction")
        for database in self.databases:
            for template in database.templates:
                
                t1 = time.time()
                imgCroped = np.asarray(template.image, dtype='int64')
                limitValues = self.getLimits(imgCroped)
                #windowsGenerated,idx,idy = self.generateWindows(imgCroped,limitValues,(self.height,self.width))

                idx,idy = self.generateSteps(limitValues,(self.height,self.width))

                newImg = {}
                for wg in imgCroped:
                    positionarrayx = [i for i in range(len(idx)) if idx[i] <= wg[0]]
                    positionarrayy = [i for i in range(len(idy)) if idy[i] <= wg[1]]
                    positionarrayx = positionarrayx[-1] if positionarrayx[-1] < self.width else self.width-1
                    positionarrayy = positionarrayy[-1] if positionarrayy[-1] < self.height else self.height-1
                    #newImg[positionarrayx][positionarrayy] = int(self.generatePixelValue(wg))
                    if not (positionarrayx in newImg.keys()):
                        newImg[positionarrayx] = {}

                    if not (positionarrayy in newImg[positionarrayx].keys()):
                        newImg[positionarrayx][positionarrayy] = []

                    newImg[positionarrayx][positionarrayy].append(wg)

                imgCroped = self.normalizeImage(newImg,0,255)

                imSave = im.fromarray(imgCroped).convert('RGB').rotate(90)
                fullPath = template.rawRepr.split(os.path.sep)
                fullPath = fullPath[-1].split('.')
                fullPath = fullPath[0]
                imSave.save(os.path.join('images_repr_cloudp',fullPath + '.bmp'))

                t2 = time.time()
                print(t2 - t1)

                #if template.layersChar is None:
                #template.layersChar = np.zeros((imgCroped.shape[0],imgCroped.shape[1],4))

                testList = [None,None,None,None]

                offsetx = int(math.ceil(imgCroped.shape[0] / float(self.windowSize)))
                offsety = int(math.ceil(imgCroped.shape[1] / float(self.windowSize)))
                fullImageDescriptor = [[],[],[],[]]
                print('Gerando descritor de: '+str(template.itemClass))
                for i in range(0,imgCroped.shape[0],offsetx):
                    lineFeatures = [None,None,None,None]
                    for j in range(0,imgCroped.shape[1],offsety):
                        desc = self.generateCode(imgCroped[i:(i+offsetx),j:(j+offsety)])
                        
                        fullImageDescriptor[0] += np.histogram(desc[0],bins=self.binsize)[0].tolist()
                        fullImageDescriptor[1] += np.histogram(desc[1],bins=self.binsize)[0].tolist()
                        fullImageDescriptor[2] += np.histogram(desc[2],bins=self.binsize)[0].tolist()
                        fullImageDescriptor[3] += np.histogram(desc[3],bins=self.binsize)[0].tolist()
                        for k in range(len(desc)):
                            if lineFeatures[k] == None:
                                lineFeatures[k] = np.array(desc[k])
                            else:
                                dCur = None
                                if (np.array(desc[k]).shape[1] != lineFeatures[k].shape[1]):
                                    dCur = self.fixSizeMatrix(np.array(desc[k]),lineFeatures[k])
                                else:
                                    dCur = np.array(desc[k])

                                lineFeatures[k] = np.concatenate((lineFeatures[k],dCur))

                    for k in range(len(desc)):
                        if testList[k] == None:
                            testList[k] = lineFeatures[k]
                        else:
                            testList[k] = np.concatenate((testList[k],lineFeatures[k]),axis=0)


                folderSave = ['x','y','w1','w2']
                for f in range(len(folderSave)):
                    template.layersChar = testList[f]
                    template.saveImageTraining(False,os.path.join('generated_images_wavelet',folderSave[f]))

                normalizedFeatures = []
                for fet in fullImageDescriptor:
                    normalizedFeatures = normalizedFeatures + (fet / np.linalg.norm(fet)).tolist()

                template.features = normalizedFeatures
                #template.features = np.array(fullImageDescriptor).flatten()
                #template.features = template.features / np.linalg.norm(template.features)
                