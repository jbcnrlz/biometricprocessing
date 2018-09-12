from baseClasses.PreProcessingStep import *
from helper.functions import getRotationMatrix3D
import math, numpy as np,matlab.engine, os

class NormalizaImageDepth(PreProcessingStep):

    def imageNormalization(self,face):
        newface = np.zeros(face.shape)
        gapBetween = abs(round(np.amax(face)) - 255)
        for i in range(face.shape[0]):
            for j in range(face.shape[1]):
                newface[i][j] = round( face[i][j] + gapBetween )
        return newface.tolist()

    def doPreProcessing(self,template):
        template.image = self.imageNormalization(np.array(template.image))
        return template
