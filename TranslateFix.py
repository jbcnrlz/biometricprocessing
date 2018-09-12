from baseClasses.PreProcessingStep import *
import math
import numpy as np

class TranslateFix(PreProcessingStep):

    def translateToOriginByNoseTip(self,points,nosetip):
        translateMatrix = np.array([[1,0,0,-nosetip[0]],[0,1,0,-nosetip[1]],[0,0,1,-nosetip[2]],[0,0,0,1]])
        for idx in range(len(points)):
            points[idx] = translateMatrix.dot(np.array(points[idx] + [1])).tolist()[:3]

        return points

    def doPreProcessing(self,template):
        template.image = self.translateToOriginByNoseTip(template.image,template.faceMarks[2])
        template.faceMarks = self.translateToOriginByNoseTip(template.faceMarks,template.faceMarks[2])
        return template

if __name__ == '__main__':
    a = [[-37.824600,51.693600,-726.000000],[26.659599,55.858101,-731.000000],[-2.407010,24.070200,-693.000000]]
    tf = TranslateFix(None)
    aline = tf.translateToOriginByNoseTip(a,[-2.407010,24.070200,-693.000000])
    print(aline)
