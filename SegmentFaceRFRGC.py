from baseClasses.PreProcessingStep import *
import math, numpy as np, pcl, os
from scipy.spatial.distance import euclidean
from helper.functions import outputObj

class SegmentFaceRFRGC(PreProcessingStep):

    def doPreProcessing(self,template):
        distancePoints = int(euclidean(template.faceMarks[0],template.faceMarks[2]))
        startingX = template.faceMarks[0][0] - distancePoints
        endingX = template.faceMarks[0][0] + distancePoints
        startingY = template.faceMarks[0][1] - distancePoints
        endingY = template.faceMarks[0][1] + distancePoints
        template.image = template.image[startingX:endingX,startingY:endingY]
        template.save(True)
        return template