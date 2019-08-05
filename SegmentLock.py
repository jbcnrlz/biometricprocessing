from baseClasses.PreProcessingStep import *
import math, numpy as np, pcl, os
from scipy.spatial.distance import euclidean
from helper.functions import outputObj
from BUTemplate import *

class SegmentLock(PreProcessingStep):

    def __init__(self,**kwargs):
        self.nosetipindex = kwargs.get('nosetip',2)
        self.distance = kwargs.get('distance', 70)
        if type(self.nosetipindex) is not int and self.nosetipindex.isdigit():
            self.nosetipindex = int(self.nosetipindex)

    def getFaceFromCenterPoint(self,center,radius,points):
        centerPoint = points[center]
        neighbors = []
        for x in points:
            distancePoints = euclidean(centerPoint,x)
            if distancePoints <= radius and x[2] != 0:
                neighbors.append(x)

        return np.array(neighbors)

    def doPreProcessing(self,template):
        distancePoints = self.distance
        imFile = None
        if type(template.image) is list:
            imFile = np.array(template.image)
        else:
            imFile = template.image

        nosetipIndex = (template.faceMarks[self.nosetipindex][0]*512) + template.faceMarks[self.nosetipindex][1]
        template.faceMarks[self.nosetipindex].append(imFile[nosetipIndex][-1])
        template.image = self.getFaceFromCenterPoint(nosetipIndex,distancePoints,imFile)

        return template