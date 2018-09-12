from baseClasses.PreProcessingStep import *
from helper.functions import findPointIndex, outputObj
import math, numpy as np, os
from subprocess import check_output
import pcl

class GeneratePCDLFW(PreProcessingStep):

    def getLimits(self,fullFace,dim=2):
        idxsB = [0] * dim
        idxsS = [0] * dim
        smallerValues = [float('inf')] * dim
        biggerValues = [-float('inf')] * dim
        for i in range(dim):
            for f in range(len(fullFace)):
                if (fullFace[f][i] < smallerValues[i]):
                    smallerValues[i] = fullFace[f][i]
                    idxsS[i] = f

                if (fullFace[f][i] > biggerValues[i]):
                    biggerValues[i] = fullFace[f][i]
                    idxsB[i] = f

        return (smallerValues + biggerValues, idxsS + idxsB)

    def doPreProcessing(self,template):
        txtFilePath = template.rawRepr[0:-4] + '.pcd'
        output = check_output('pcl_mesh2pcd ' + template.rawRepr + ' ' + txtFilePath+ ' -no_vis_result',shell=True)
        p = pcl.load(txtFilePath)
        template.image = np.array(p.to_array())
        limits = self.getLimits(template.image,3)
        '''
        pointX = (abs(limits[0]) - abs(limits[3]))/2.0
        if (limits[0] < 0):
            pointX = limits[0] - pointX
        else:
            pointX = limits[0] + pointX            
        pointY = (abs(limits[1]) - abs(limits[4]))/2.0
        if (limits[2] < 0):
            pointX = limits[1] - pointX
        else:
            pointX = limits[1] + pointX

        pidx = findPointIndex(template.image,np.array([pointX,pointY,limits[2]]))
        '''
        template.image = template.image - template.image[limits[1][5]]
        cloudp = pcl.PointCloud()
        cloudp.from_array(template.image)
        os.remove(txtFilePath)
        cloudp.to_file(txtFilePath.encode('utf-8'))
        return template