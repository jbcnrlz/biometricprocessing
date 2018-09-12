import numpy as np, os, pcl, math, operator,datetime
from baseClasses.BiometricProcessing import *
from sklearn import svm
from PIL import Image as im
from scipy.spatial.distance import *
from helper.functions import minmax, generateHistogram, loadOBJ, getRegionFromCenterPoint, findPointIndex, outputObj, printProgressBar
from random import randint, sample
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class acdn(BiometricProcessing):

    def __init__(self,database):
        self.binsize = 14
        self.databases = database
        self.methodName = 'ACDNP'
        super().__init__()

    def findCharA(self,pointA,pointB):
        u = np.array(pointA)
        v = np.array(pointB)
        return np.arccos(np.dot(u,v) / ( math.sqrt(sum([i**2 for i in u])) * math.sqrt(sum([i**2 for i in v])) ) )

    def findCharC(self,pointA,pointB,noseTip):
        AB = math.sqrt( (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2 + (pointA[2] - pointB[2])**2)
        AN = math.sqrt( (pointA[0] - noseTip[0])**2 + (pointA[1] - noseTip[1])**2 + (pointA[2] - noseTip[2])**2)
        BN = math.sqrt( (pointB[0] - noseTip[0])**2 + (pointB[1] - noseTip[1])**2 + (pointB[2] - noseTip[2])**2)
        s = (AB + AN + BN) / 2
        area = s*(s - AB)*(s - AN) *(s - BN)
        if (area == 0):
            return 0
        else:
            return (AB * AN * BN) / (4 * area)

    def findCharD(self,pointA,pointB):
        return math.sqrt( (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2 + (pointA[2] - pointB[2])**2)

    def findCharN(self,pointA,pointB):
        pointTranslated = np.array(pointA) + (np.array(pointB) * -1)
        return self.findCharA(pointTranslated,np.array([0,0,1]))

    def findCharP(self,pointA,pointB):
        return self.findCharA(pointA,np.array([0,0,1])), self.findCharA(pointB,np.array([0,0,1]))

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

    def generateIndexSamples(self,s1,s2):
        returnList = []
        for i in range(len(s1)):
            if s1[i] == s2[i] and i < len(s2)-1:
                tmp = s2[i]
                s2[i] = s2[i+1]
                s2[i+1] = tmp

            returnList.append((s1[i],s2[i]))
        return returnList

    def featureExtraction(self):
        for database in self.databases:
            for template in database.templates:
                tmpIniTotal = datetime.datetime.now()
                template.features = []
                print("========")
                print("Fazendo do " + str(template.itemClass))

                centralPoints = None
                limits = 40
                if template.faceMarks:
                    centralPoints = template.faceMarks[:3]
                else:
                    limits = 100000000
                    centralPoints = np.array([[0,0,0]])

                for center in centralPoints:
                    tini = datetime.datetime.now()
                    idxCenter = findPointIndex(template.image,center)
                    tfim = datetime.datetime.now()
                    r = tfim - tini
                    print("Total point index= "+str(r))
                    tini = datetime.datetime.now()
                    currRegion = getRegionFromCenterPoint(idxCenter,limits,template.image)
                    tfim = datetime.datetime.now()
                    r = tfim - tini
                    print("Total center point= "+str(r))
                    samplesSize = round(len(currRegion) * 0.8)
                    jaforam = []
                    chara = []
                    charc = []
                    chard = []
                    charn = []
                    charp = [[],[]]
                    samples = sample(list(range(len(currRegion))),samplesSize)
                    samples2 = sample(list(range(len(currRegion))),samplesSize)
                    listFixed = self.generateIndexSamples(samples,samples2)
                    step = 0                    
                    for l in listFixed:
                        step += 1
                        print('\r%s de %s' % (str(step), str(len(samples))), end = '\r')
                        chara.append(self.findCharA(currRegion[l[0]],currRegion[l[1]]))
                        charc.append(self.findCharC(currRegion[l[0]],currRegion[l[1]],np.array([0,0,0])))
                        chard.append(self.findCharD(currRegion[l[0]],currRegion[l[1]]))
                        charn.append(self.findCharN(currRegion[l[0]],currRegion[l[1]]))
                        resultsP = self.findCharP(currRegion[l[0]],currRegion[l[1]])
                        charp[0].append(resultsP[0])
                        charp[1].append(resultsP[1])
                    template.features += generateHistogram(chara,self.binsize) + generateHistogram(charc,self.binsize) + generateHistogram(chard,self.binsize) + generateHistogram(charn,self.binsize) + generateHistogram(charp[1],self.binsize)                
                    '''
                    while len(jaforam) < samples:
                        p1 = randint(0, len(currRegion) - 1)
                        p2 = randint(0, len(currRegion) - 1)
                        while (jaforam.count((p1,p2)) != 0 or jaforam.count((p2,p1)) != 0) and p1 == p2:
                            p1 = randint(0, len(currRegion) - 1)
                            p2 = randint(0, len(currRegion) - 1)
                        jaforam.append((p1,p2))
                        chara.append(self.findCharA(currRegion[p1],currRegion[p2]))
                        charc.append(self.findCharC(currRegion[p1],currRegion[p2],np.array([0,0,0])))
                        chard.append(self.findCharD(currRegion[p1],currRegion[p2]))
                        charn.append(self.findCharN(currRegion[p1],currRegion[p2]))
                        resultsP = self.findCharP(currRegion[p1],currRegion[p2])
                        charp[0].append(resultsP[0])
                        charp[1].append(resultsP[1])
                    template.features += generateHistogram(chara,self.binsize) + generateHistogram(charc,self.binsize) + generateHistogram(chard,self.binsize) + generateHistogram(charn,self.binsize) + generateHistogram(charp[1],self.binsize)
                    '''
                template.image = None
                self.saveFeature(template)
                tmpFiniTotal = datetime.datetime.now()
                r = tmpFiniTotal - tmpIniTotal
                print("Total sujeito = "+str(r))

    def loadRotatedFaces(self,angles):
        addTemplates = []
        for t in self.templates:
            print("Gerando copia de "+t.rawRepr)
            for a in angles:
                print("Angulo = "+ str(a))
                nobj = copy.deepcopy(t)
                with im.open(nobj.rawRepr[0:-4] + '_rotate_'+str(a)+'_newdepth.bmp') as currImg:
                    nobj.image = np.asarray(currImg)
                    nobj.rawRepr = nobj.rawRepr[0:-4] + '_rotate_'+str(a)+'_newdepth.bmp'
                addTemplates.append(nobj)

    def matcher(self,feedProba=False):
        chutes = np.zeros(52)
        gal = self.databases[0].getDatabaseRepresentation()
        probe = self.databases[1].getDatabaseRepresentation()

        clf = svm.SVC(gamma=1e-3, C=100,probability=True)
        clf.fit(np.array(gal[0]),np.array(gal[1]))
        pred = clf.predict(probe[0])
        print(pred)
        print(classification_report(probe[1], pred))
        print(confusion_matrix(probe[1], pred, labels=range(52)))
        input()
        '''
        tuned_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}

        clf = svm.SVC(gamma=1e-3, C=100,probability=True)
        clf.fit(np.array(gal[0]),np.array(gal[1]))
        resultados = [0,0]
        for itemNumber in range(len(probe[0])):
            if (feedProba):
                self.feedProbilityTemplate(probe[1][itemNumber],clf.predict_proba(probe[0][itemNumber]).tolist().pop())
            classAchada = clf.predict(probe[0][itemNumber]).tolist().pop()
            chutes[classAchada - 1] += 1
            resultados[classAchada == probe[1][itemNumber]] += 1
        '''
        return pred