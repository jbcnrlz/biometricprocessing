from baseClasses.PreProcessingStep import *
import math, numpy as np, pcl, os
from scipy.spatial.distance import euclidean
from helper.functions import outputObj

class SegmentFace(PreProcessingStep):

    def __init__(self,**kwargs):
        self.nosetipindex = kwargs.get('nosetip',2)
        if type(self.nosetipindex) is not int and self.nosetipindex.isdigit():
            self.nosetipindex = int(self.nosetipindex)

    def findCenterIndex(self,points,center):
        smallerDistance = [100000000000000000000000000000000000000000000000000000000,0]
        for p in range(len(points)):
            if (points[p][0] == center[0] and points[p][1] == center[1] and points[p][2] == center[2]):
                return p
            elif(euclidean(np.array(points[p]),np.array(center)) < smallerDistance[0]):
                smallerDistance[0] = euclidean(np.array(points[p]),np.array(center))
                smallerDistance[1] = p
        else:
            return smallerDistance[1]

    def getFaceFromCenterPoint(self,center,radius,points):
        center = self.findCenterIndex(points,center)
        centerPoint = points[center]
        sortedList = list(points)
        neighbors = []
        for x in range(len(sortedList)):
            distancePoints = euclidean(np.array(centerPoint[0:2]),np.array(sortedList[x][0:2]))
            if distancePoints <= radius:
                neighbors.append(sortedList[x])
        '''
        for x in range(center + 1,len(sortedList)):
            if ((centerPoint != sortedList[x]).all() and (sortedList[x] != [0.0,0.0,0.0]).all()):
                distancePoints = euclidean(np.array(centerPoint[0:2]),np.array(sortedList[x][0:2]))
                if distancePoints <= radius:
                    neighbors.append(sortedList[x])
        '''
        return neighbors

    def doPreProcessing(self,template):
        distancePoints = 70
        imFile = None
        if type(template.image) is list:
            imFile = np.array(template.image)
        else:
            imFile = template.image
        if len(template.faceMarks) > 0:
            template.image = self.getFaceFromCenterPoint(template.faceMarks[self.nosetipindex],distancePoints,imFile)
        else:
            template.image = self.getFaceFromCenterPoint(np.array([0,0,0]),60000,imFile)
            txtFilePath = template.rawRepr[0:-4] + '_segmented.pcd'
            cloudp = pcl.PointCloud()
            cloudp.from_array(np.array(template.image))
            if (os.path.exists(txtFilePath)):
                os.remove(txtFilePath)
            cloudp.to_file(txtFilePath.encode('utf-8'))

        return template