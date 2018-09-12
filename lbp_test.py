import numpy as np, math, operator, os, logging, scipy.ndimage as ndimage, matlab.engine
from baseClasses.BiometricProcessing import *
from helper.functions import generateHistogram, loadOBJ
from sklearn import svm
from helper.functions import minmax
from PIL import Image as im
from scipy.spatial.distance import *
import mahotas

class LBPMahotas(BiometricProcessing):

    def __init__(self,windowsize,database):
        self.windowSize = windowsize
        self.databases = database

    def generateImageDescriptor(self,image,p=8,r=1):
        return mahotas.features.lbp(image,r,p).tolist()

    def cropImage(self,image,le,re,no):
        distance = list(map(operator.sub,le[:2],re[:2]))
        distance = int(math.sqrt((distance[0]**2) + (distance[1]**2)))
        points_crop = (no[0] - distance,no[1] - distance,no[0] + distance,no[1] + distance)
        points_crop = list(map(int,points_crop))   
        return image.crop((points_crop[0],points_crop[1],points_crop[2],points_crop[3]))

    def preProcessing(self):
        for database in self.databases:
            for template in database.templates:
                print('Gerando pre processamento de: ' + str(template.itemClass) + ' ' + str(database.folderTemplate))
                if (template.faceMarks != []):                    
                    template.image = self.cropImage(template.image,template.faceMarks[0],template.faceMarks[1],template.faceMarks[2]).resize((32,32),im.BILINEAR)
        
    def featureExtraction(self):
        for database in self.databases:
            for template in database.templates:
                imgCroped = np.asarray(template.image)                
                print('Gerando descritor de: '+str(template.itemClass))
                template.features = self.generateImageDescriptor(imgCroped)

    def matcher(self):
        chutes = np.zeros(52)
        gal = self.databases[0].getDatabaseRepresentation()
        probe = self.databases[1].getDatabaseRepresentation()

        clf = svm.SVC(gamma=0.001, C=100.)
        clf.fit(gal[0],gal[1])
        resultados = [0,0]
        for itemNumber in range(len(probe[0])):
            classAchada = clf.predict(probe[0][itemNumber]).tolist().pop()
            chutes[classAchada - 1] += classAchada == probe[1][itemNumber]
            resultados[classAchada == probe[1][itemNumber]] += 1

        print(chutes)
        return resultados
