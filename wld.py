import numpy as np, math, operator, os, logging, scipy.ndimage as ndimage, matlab.engine
from baseClasses.BiometricProcessing import *
from helper.functions import generateHistogram, loadOBJ, generateHistogramUniform, bilinear_interpolation, mergeArraysDiff
from sklearn import svm
from helper.functions import minmax
from PIL import Image as im
from scipy.spatial.distance import *
from scipy.special import expit
from mahotas.interpolate import shift

class WLD(BiometricProcessing):

    def __init__(self,C,T,database):
        self.databases = database
        self.C = C
        self.T = T

    def generateImageDescriptor(self,image,p=8,r=1):
        returnValue = [[],[],[],[]]
        #returnValue = self.generateCodePR(image,[1,1],p,r)

        log= logging.getLogger( "ThreeDLBP.generateImageDescriptor" )
        #log.debug("Valor de retorno"+str(returnValue))

        
        for i in range(r,image.shape[0]-r):
            for j in range(r,image.shape[1]-r):
        #for i in range(1,image.shape[0]-1):
        #    for j in range(1,image.shape[1]-1):
                #resultCode = self.generateCodePR(image[i-r:i+(1+r),j-r:j+(1+r)],np.array([r,r]),p,r,'Sigmoide')
                #resultCode = self.generateCodePR(np.array([[1,2,3],[4,5,6],[7,8,9]]),np.array([r,r]),p,r)
                #resultCode = self.generateCode(image[i-r:i+(1+r),j-r:j+(1+r)],np.array([1,1]))
                resultCode = self.generateCode(image[i-1:i+2,j-1:j+2],np.array([1,1]))
                returnValue[0].append(resultCode[0])
                returnValue[1].append(resultCode[1])
                returnValue[2].append(resultCode[2])
                returnValue[3].append(resultCode[3])
        
        return returnValue

    def cropImage(self,image,le,re,no):
        distance = list(map(operator.sub,le[:2],re[:2]))
        distance = int(math.sqrt((distance[0]**2) + (distance[1]**2)))
        points_crop = (no[0] - distance,no[1] - distance,no[0] + distance,no[1] + distance)
        points_crop = list(map(int,points_crop))   
        return image.crop((points_crop[0],points_crop[1],points_crop[2],points_crop[3]))

    def setupTemplate(self,template):
        return template

    def cleanupTemplate(self,template):
        return template

    def extractRegion(self,image,centerPosition):
        if (type(image) is not np.ndarray):
            image = np.array(image)
        return image[centerPosition[0]-1:centerPosition[0]+2,centerPosition[1]-1:centerPosition[1]+2]

    def calculateDiffExcit(self,region):
        if (type(region) is not np.ndarray):
            region = np.array(region)

        diffEx = 0
        centerPixel = 1
        if region[centerPixel,centerPixel] != 0:
            for i in range(region.shape[0]):
                for j in range(region.shape[1]):
                    diffEx += region[i,j] - region[centerPixel,centerPixel]

            diffEx = diffEx / region[centerPixel,centerPixel]

        return np.arctan(diffEx)

    def getThetaLine(self,region):
        theta = 0
        v11s = region[1,0] - region[1,2]
        v10s = region[2,1] - region[0,1]
        if (v10s != 0):
            theta = np.arctan(v11s/v10s)

            atan2 = 0

            if (v11s > 0) and (v10s > 0):
                atan2 = theta
            elif (v11s > 0) and (v10s < 0):
                atan2 = np.pi + theta
            elif (v11s < 0) and (v10s < 0):
                atan2 = theta - np.pi
            elif (v11s < 0) and (v10s > 0):
                atan2 = theta
        else:
            atan2 = 0
        return atan2 + np.pi

    def quantOrientations(self,thetaLine,T):
        division = (2*np.pi) / T
        division = thetaLine / division
        t = math.fmod(math.floor( division + 0.5 ),T)
        return ((2*t) / T) * np.pi

    def generateHistogram(self,valuesExc,valuesOri):
        histCombined = np.zeros((self.C,self.T))
        for v in np.unique(valuesExc):
            idxExc = np.histogram(v,bins=self.C,range=(-np.pi/2,np.pi/2))[0]
            idxExc = np.argwhere(idxExc==1)[0][0]
            positionUniques = np.argwhere(valuesExc==v)
            for pu in positionUniques:
                posOri = np.histogram(valuesOri[pu[0],pu[1]],bins=self.T,range=(0,np.pi*2))[0]
                posOri = np.argwhere(posOri==1)[0][0]            
                histCombined[idxExc,posOri] += 1

        return histCombined

    def generateImageRepr(self,imageValues,lowerLimit=-np.pi/2,higherLimit=np.pi/2):
        for i in range(imageValues.shape[0]):
            for j in range(imageValues.shape[1]):
                histogramPost = np.histogram(imageValues[i,j],bins=255,range=(lowerLimit,higherLimit))[0]
                histogramPost = np.argwhere(histogramPost==1)[0][0]
                imageValues[i,j] = histogramPost

        return imageValues

    def featureExtraction(self,excludeFile=[]):
        for database in self.databases:
            for template in database.templates:
                imgCroped = np.asarray(template.image)

                print('Gerando descritor de: '+str(template.itemClass))
                excitation = np.zeros((imgCroped.shape[0] - 1,imgCroped.shape[1] - 1), dtype=np.float64)
                orientations = np.zeros((imgCroped.shape[0] - 1,imgCroped.shape[1] - 1), dtype=np.float64)
                for i in range(1,imgCroped.shape[0] - 1):
                    for j in range(1,imgCroped.shape[1] - 1):
                        region = self.extractRegion(imgCroped,[i,j]).astype(np.float64)
                        cde    = self.calculateDiffExcit(region)
                        orient = self.quantOrientations(self.getThetaLine(region),8)
                        excitation[i-1,j-1] = cde
                        orientations[i-1,j-1] = orient

                template.features = self.generateHistogram(excitation,orientations).flatten()
                #template.saveHistogramImage()
                #template.saveHistogramImage(self.generateImageRepr(excitation),'generated_images_wld_excitation')
                #template.saveHistogramImage(self.generateImageRepr(orientations,0,2*np.pi),'generated_images_wld_orientation')
