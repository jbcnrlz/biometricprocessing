from baseClasses.PreProcessingStep import *
import numpy as np, pcl, os

class InitialFilter(PreProcessingStep):

    def doPreProcessing(self,template):
        newImage = []
        for pixel in template.image:
            if pixel[0] > -2000 and pixel[1] > -2000 and pixel[2] > -2000:
                newImage.append(pixel)

        template.image = np.array(newImage)
        return template
