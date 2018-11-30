from baseClasses.PreProcessingStep import *
import numpy as np, pcl, os

class TranslateFix(PreProcessingStep):

    def __init__(self,**kwargs):
        self.nosetipindex = kwargs.get('nosetip',2)
        if type(self.nosetipindex) is not int and self.nosetipindex.isdigit():
            self.nosetipindex = int(self.nosetipindex)

    def translateToOriginByNoseTip(self,points,nosetip):
        translateMatrix = np.array([[1,0,0,-nosetip[0]],[0,1,0,-nosetip[1]],[0,0,1,-nosetip[2]],[0,0,0,1]])
        for idx in range(len(points)):
            homogeneous = points[idx] + [1] if type(points[idx]) is list else points[idx].tolist() + [1]
            points[idx] = translateMatrix.dot(np.array(homogeneous)).tolist()[:3]

        return points

    def doPreProcessing(self,template):
        template.image = self.translateToOriginByNoseTip(template.image,template.faceMarks[self.nosetipindex])
        if (type(self.nosetipindex) is int) or (self.nosetipindex.isdigit()):
            template.faceMarks = self.translateToOriginByNoseTip(template.faceMarks,template.faceMarks[self.nosetipindex])

        txtFilePath = template.rawRepr[0:-4] + '_segmented.pcd'
        cloudp = pcl.PointCloud()
        cloudp.from_array(np.array(template.image, dtype=np.float32))
        if (os.path.exists(txtFilePath)):
            os.remove(txtFilePath)
        cloudp.to_file(txtFilePath.encode('utf-8'))

        return template
