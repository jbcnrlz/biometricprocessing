import numpy as np, os, pcl
from baseClasses.BiometricProcessing import *
from sklearn import svm
from PIL import Image as im
from scipy.spatial.distance import *
from helper.functions import minmax, generateHistogram, loadOBJ, getRegionFromCenterPoint, findPointIndex, outputObj
from random import randint
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

class LBPCloudPoint(BiometricProcessing):

    def __init__(self,kdeNeighboursQuantity,binsize,database):
        self.binsize = binsize
        self.kdeNeighboursQuantity = kdeNeighboursQuantity + 1
        self.databases = database

    def generateIndexes(self,R,P,p):
        x = round(-R * np.sin( (2*np.pi*p) / P) + 1,5)
        y = round(R * np.cos( (2*np.pi*p) / P) + 1,5)
        return (x,y)

    def getAverage(self,region,axis):
        returnAverage = 0
        for r in region:
            returnAverage += r[axis]

        returnAverage = returnAverage / float(len(region))
        return returnAverage

    def generateCode(self,template):
        codes = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]        
        for x in range(len(template.image)):
            windowsArray = [np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))]
            template.createGraph(x,self.kdeNeighboursQuantity)
            #averageLocal = [0,0,0]
            #xAvgNeighs = [[],[],[]]

            for chan in range(3):
                windowsArray[chan][1][1] = self.getAverage(template.graph.point,chan)
                idxsZMatrix = [0,0]
                areas = template.graph.neighbours
                for n in template.graph.neighbours:
                    
                    windowsArray[chan][idxsZMatrix[0]][idxsZMatrix[1]] = self.getAverage(n,chan)
                    idxsZMatrix[0] += 1
                    if (idxsZMatrix[0] > 2):
                        idxsZMatrix[0] = 0
                        idxsZMatrix[1]+= 1
                    elif (idxsZMatrix[0] == 1 and idxsZMatrix[1] == 1):
                        idxsZMatrix[0] += 1

                code = self.generateCodeLBP(windowsArray[chan],np.array([1,1]))
                codes[chan][0].append(code[0])
                codes[chan][1].append(code[1])
                codes[chan][2].append(code[2])
                codes[chan][3].append(code[3])
        return codes

    def generateCodeLBP(self,image,center):
        idxs = [
            (center[0]-1,center[1]-1),
            (center[0]-1,center[1]),
            (center[0]-1,center[1]+1),
            (center[0],center[1]-1),
            (center[0],center[1]+1),
            (center[0]+1,center[1]-1),
            (center[0]+1,center[1]),
            (center[0]+1,center[1]+1)
        ]
        layers = [[],[],[],[]]
        for i in idxs:
            subraction = int(round(image[int(i[0])][int(i[1])] - image[center[0]][center[1]]))
            if subraction < -7:
                subraction = -7
            elif subraction > 7:
                subraction = 7
            bin = '{0:03b}'.format(abs(subraction))
            layers[0].append(str(int(subraction >= 0)))
            layers[1].append(bin[0])
            layers[2].append(bin[1])
            layers[3].append(bin[2])
        for l in range(len(layers)):
            layers[l] = int(''.join(layers[l]),2)
        return layers


    def setupTemplate(self,template):
        template.loadMarks('3DObj')
        subject = "%04d" % (template.itemClass)
        template3Dobj = template.rawRepr.split(os.path.sep)[:-3]
        folderType = template3Dobj[template3Dobj.index(subject) + 1]
        a, b, c, y = loadOBJ(os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'.obj'))
        template.image = c        
        return template

    def cleanupTemplate(self,template):        
        #template.image = im.fromarray(np.array(template.image,dtype=np.uint8))
        return template

    def featureExtraction(self):
        for database in self.databases:
            for template in database.templates:
                print("========")
                print("Fazendo do " + str(template.itemClass))
                originalFace = template.image
                allFeatures = []

                for center in template.faceMarks[:3]:
                    idxCenter = findPointIndex(originalFace,center)
                    template.image = getRegionFromCenterPoint(idxCenter,40,originalFace)
                    allFeatures.append(self.generateCode(template))

                template.features = []
                for feat in allFeatures:
                    for f in feat:
                        template.features += generateHistogram(f[0],self.binsize) + generateHistogram(f[1],self.binsize) + generateHistogram(f[2],self.binsize) + generateHistogram(f[3],self.binsize)

    def matcher(self):
        chutes = np.zeros(52)
        gal = self.databases[0].getDatabaseRepresentation()
        probe = self.databases[1].getDatabaseRepresentation()

        tuned_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}

        
        for g in tuned_parameters['gamma']:
            for c in tuned_parameters['C']:
                clf = svm.SVC(gamma=g, C=c)
                clf.fit(np.array(gal[0]),np.array(gal[1]))
                resultados = [0,0]        
                for itemNumber in range(len(probe[0])):
                    classAchada = clf.predict(probe[0][itemNumber]).tolist().pop()
                    chutes[classAchada - 1] += 1
                    resultados[classAchada == probe[1][itemNumber]] += 1

                print(str(g) + ' ' + str(c))
                print(resultados)
                input()

        return []
        #return resultados
        
        

