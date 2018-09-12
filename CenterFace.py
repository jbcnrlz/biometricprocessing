from scipy.spatial.distance import euclidean
from baseClasses.PreProcessingStep import *

class CenterFace(PreProcessingStep):

    def findBiggerValue(self,points):
        biggerValues = -float('inf')
        idxBigger = -1
        for p in range(len(points)):
            if (points[p][2] > biggerValues):
                biggerValues = points[p][2]
                idxBigger = p

        return idxBigger

    def doPreProcessing(self,template):
        bigger = self.findBiggerValue(template.image)
        template.image = bigger - template.image
        template.save(True,'_centered')
        return template