from baseClasses.PreProcessingStep import *
import numpy as np

class FixPaperOcclusion(PreProcessingStep):

    def removeNoise(self,face):
        newface = []
        for f in face:
            if f[0] >= 0 and f[2] <= 0:
                newface.append(f)

        return np.array(newface)

    def doPreProcessing(self,template):
        if (template.typeTemplate == 'OcclusionPaper'):
            template.image = self.removeNoise(template.image)
            return template