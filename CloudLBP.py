import numpy as np, math, operator, os, logging, scipy.ndimage as ndimage, matlab.engine, pywt
from baseClasses.BiometricProcessing import *
from helper.functions import generateHistogram, loadOBJ, generateHistogramUniform, bilinear_interpolation, mergeArraysDiff, minmax
from PIL import Image as im
from scipy.spatial.distance import euclidean
from scipy.special import expit
from mahotas.interpolate import shift
from LFWTemplate import *
from models.models import *
from models.engine_creation import *

class CloudLBP(BiometricProcessing):

    def __init__(self,windowsize,binsize,database,inCircleMaxItens=8,outCircleMaxItens=40,slices=8,radius=5,sliceCenter=[0,0],samp=0.4):
        self.windowSize = windowsize
        self.binsize    = binsize
        self.databases  = database
        self.methodName = 'CloudLBP'
        self.innerCircleLimit = inCircleMaxItens
        self.outerCircleLimit = outCircleMaxItens
        self.sliceQuant  = slices
        self.radiusSlice = radius
        self.sliceCenter = sliceCenter
        self.sampleSize  = samp
        super().__init__()

    def generateCode(self,slices,avgCenter,typeOp='Normal'):
        layers = [[],[],[],[]]
        for s in slices:
            subNeigh = int(round(s - avgCenter))
            if subNeigh < -7:
                subNeigh = -7
            elif subNeigh > 7:
                subNeigh = 7

            binNumber = '{0:03b}'.format(abs(subNeigh))
            layers[0].append(str(int(subNeigh >= 0)))
            layers[1].append(binNumber[0])
            layers[2].append(binNumber[1])
            layers[3].append(binNumber[2])

        for l in range(len(layers)):
            layers[l] = int(''.join(layers[l]),2)

        return layers
                    

    def generateImageDescriptor(self,image,p=8,r=1,typeLBP='original',typeMeasurement='Normal'):
        returnValue = [[],[],[],[]]
        for i in range(r,image.shape[0]-r):
            for j in range(r,image.shape[1]-r):
                if typeLBP == 'original':
                    resultCode = self.generateCode(image[i-1:i+2,j-1:j+2],np.array([1,1]),typeMeasurement)
                elif typeLBP == 'pr':
                    resultCode = self.generateCodePR(image[i-r:i+(r+1),j-r:j+(r+1)],np.array([r,r]),p,r,typeMeasurement)
                returnValue[0].append(resultCode[0])
                returnValue[1].append(resultCode[1])
                returnValue[2].append(resultCode[2])
                returnValue[3].append(resultCode[3])
        
        return returnValue

    def setupTemplate(self,template):
        if (not type(template) is LFWTemplate):
            template.loadMarks('3DObj')
            subject = "%04d" % (template.itemClass)
            template3Dobj = template.rawRepr.split(os.path.sep)[:-3]
            folderType = template3Dobj[template3Dobj.index(subject) + 1]
            a, b, c, y = loadOBJ(os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'.obj'))
            template.image = c

        return template

    def cleanupTemplate(self,template):
        template.layersChar = np.zeros((len(template.image),len(template.image[0]),4))
        template.image = im.fromarray(np.array(template.image,dtype=np.uint8))
        template.image = template.image.rotate(-180)
        template.save(True)
        return template

    def generateInnerOuter(self,imageFace,centerPoint):
        neigh = []
        centerNeigh = [imageFace[centerPoint]]
        idxUp = centerPoint - 1
        idxDown = centerPoint + 1
        while True:            
            if ((self.innerCircleLimit <= len(centerNeigh)) and (self.outerCircleLimit <= len(neigh))) or ((idxUp <= 0) and (idxDown >= len(imageFace))):
                break

            if (idxUp >= 0):
                dist = euclidean(imageFace[centerPoint],imageFace[idxUp])
                if (dist < 5) and (dist >= 2) and (self.outerCircleLimit > len(neigh)):
                    neigh.append(imageFace[idxUp])
                elif (dist < 2) and (self.innerCircleLimit > len(centerNeigh)):
                    centerNeigh.append(imageFace[idxUp])
                idxUp -= 1
            
            if (idxDown < len(imageFace)):
                dist = euclidean(imageFace[centerPoint],imageFace[idxDown])
                if (dist < 5) and (dist >= 2) and (self.outerCircleLimit > len(neigh)):
                    neigh.append(imageFace[idxDown])
                elif (dist < 2) and (self.innerCircleLimit > len(centerNeigh)):
                    centerNeigh.append(imageFace[idxDown])        
                idxDown += 1

        return np.array(neigh), np.array(centerNeigh)

    def areClockwise(self,v1, v2):
      return ((-v1[0]*v2[1] + v1[1]*v2[0]) > 0)

    def getSlices(self,cp,n,r):
        tet = np.linspace(-math.pi,math.pi,n+1)
        xi = r * np.cos(tet)+cp[0]
        yi = r * np.sin(tet)+cp[1]

        return xi, yi

    def getAvgNeigh(self,neigh,chan=2):
        return np.average(neigh[:,chan])

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
        #fullFace.sort(0)
        while(currentY <= limits[3]):
            idySep.append(currentY)
            currentX = limits[0]
            while(currentX <= limits[2]):
                #print('\r Gerando Janela X %03d Y %03d' % (currentX,currentY), end = '\r')
                if not currentX in idxSep:
                    idxSep.append(currentX)
                region.append([])
                deleteFromFace = []
                for f in range(len(fullFace)):
                    if (fullFace[f][0] >= currentX) and (fullFace[f][0] < currentX + windowXSize) and (fullFace[f][1] >= currentY) and (fullFace[f][1] < currentY + windowYSize):
                        deleteFromFace.append(f)
                        region[len(region) - 1].append(fullFace[f])
                    #elif (len(fullFace) > 0):
                    #    break

                fullFace = np.delete(fullFace,deleteFromFace,0)
                currentX += windowXSize
            currentY += windowYSize

        return region, idxSep, idySep

    def featureExtraction(self,excludeFile=[]):        
        print("Iniciando feature extraction")
        for database in self.databases:            
            for template in database.templates:
                print('Gerando descritor de: '+str(template.itemClass))
                fullImageDescriptor = []
                imgCroped = np.asarray(template.image)

                limitValues = self.getLimits(imgCroped)
                windowsGenerated,idx,idy = self.generateWindows(imgCroped,limitValues,(8,8))

                for w in windowsGenerated:

                    imgCroped = np.array(w)
                    imgCroped.sort(0)
                    indexes = [i for i in range(len(imgCroped))]
                    indexes = random.sample(indexes,int(len(imgCroped)*self.sampleSize))
                    xi, yi = self.getSlices(self.sliceCenter,self.sliceQuant,self.radiusSlice)
                    desc = [[],[],[],[]]
                    for idxIF in indexes:
                        neigh, centerNeigh = self.generateInnerOuter(imgCroped,idxIF)
                        if (len(neigh)) == 0:
                            desc[0].append(0)
                            desc[1].append(0)
                            desc[2].append(0)
                            desc[3].append(0)
                            continue

                        neigh = neigh - imgCroped[idxIF]
                        centerNeigh = centerNeigh - imgCroped[idxIF]

                        slices = []
                        for idx in range(xi.shape[0] - 1):
                            startSec = [xi[idx],yi[idx]]
                            endSec = [xi[idx+1],yi[idx+1]]
                            slices.append([])            
                            for n in neigh:
                                if (not self.areClockwise(startSec,n[:-1])) and (self.areClockwise(endSec,n[:-1])):
                                    slices[-1].append(n)
                            if (len(slices[-1]) == 0):
                                slices[-1] = 0
                            else:
                                slices[-1] = self.getAvgNeigh(np.array(slices[-1]))

                        avgCenter = self.getAvgNeigh(centerNeigh)

                        layers = self.generateCode(slices,avgCenter)
                        if (len(layers) == 4):
                            desc[0].append(layers[0])
                            desc[1].append(layers[1])
                            desc[2].append(layers[2])
                            desc[3].append(layers[3])
                        else:
                            desc[0].append(0)
                            desc[1].append(0)
                            desc[2].append(0)
                            desc[3].append(0)

                    fullImageDescriptor += generateHistogram(desc[0],self.binsize) + generateHistogram(desc[1],self.binsize) + generateHistogram(desc[2],self.binsize) + generateHistogram(desc[3],self.binsize)

                template.features = fullImageDescriptor
