from baseClasses.PreProcessingStep import *
from GenerateNewDepthMaps import *
import math
import numpy as np
import copy
from PIL import Image as im

class RotateFace(PreProcessingStep):

    def __init__(self,**kwargs):
        self.axis = kwargs.get('axis', ['x','y','xy'])

    def outputObj(self,points,fileName):
        f = open(fileName,'w')
        f.write('g \n')
        for p in points:
            if (len(p) == 2):
                f.write('v ' + ' '.join(map(str,p[0])) + '\n')
            else:
                f.write('v ' + ' '.join(map(str,p)) + '\n')
        f.write('g 1 \n')
        f.close()

    def getRotationMatrix(self,angle,matrix):
        cosResult = np.cos(np.radians(angle))
        sinResult = np.sin(np.radians(angle))
        if matrix == 'x':
            return np.array([
                        (1 ,0, 0, 0),
                        (0 , cosResult, -sinResult, 0),                
                        (0 , sinResult,  cosResult, 0),
                        (0 ,         0,          0, 1)
                    ])
        elif matrix == 'y':
            return np.array([
                        (cosResult , 0, sinResult, 0),
                        (0         , 1, 0        , 0),
                        (-sinResult, 0, cosResult, 0),
                        (0         , 0,         0, 1)                
                    ])
        else:
            return np.array([
                        (cosResult ,-sinResult, 0, 0),
                        (sinResult , cosResult, 0, 0),                
                        (0         ,         0, 1, 0),
                        (0         ,         0, 0, 1)
                    ])

    def multiplyMatrices(self,m1,m2):
        multiResult = []
        for m in m1:
            multMat = np.dot(m+[1],m2)[:-1]
            multiResult.append(multMat)

        return np.array(multiResult)

    def doPreProcessing(self,template):
        genFaces = GenerateNewDepthMaps()
        faceCloud = template.image
        #pathFile = template.rawRepr.split(os.path.sep)
        if (not os.path.exists(os.path.join('temporaryTemplate','objRotated'))):
            os.makedirs(os.path.join('temporaryTemplate','objRotated'))

        for ax in self.axis:
            for i in range(-30,31,10):
                if (i != 0):
                    nObj = copy.deepcopy(template)
                    if ax == 'xy':
                        rtx = self.getRotationMatrix(i, 'x')
                        for j in range(-30,31,10):
                            rty = self.getRotationMatrix(j, 'y')
                            nObj.image = self.multiplyMatrices(faceCloud, rtx)
                            nObj.image = self.multiplyMatrices(nObj.image.tolist(), rty)
                            self.outputObj(nObj.image,os.path.join('temporaryTemplate','objRotated',nObj.rawRepr[0:-4] + '_rotate_'+str(i)+'_'+str(j)+'_'+ax+'_newdepth.obj'))
                            nObj = genFaces.doPreProcessing(nObj)
                            nObj.image = im.fromarray(np.array(nObj.image,dtype=np.uint8))
                            nObj.image = nObj.image.rotate(-180)
                            pathCImg = nObj.rawRepr.split(os.path.sep)
                            if pathCImg.index('EURECOM_Kinect_Face_Dataset') >=0 :
                                fileName = pathCImg[-1]
                                pathCImg = os.path.sep.join(pathCImg[0:-2])
                                nObj.image.save(os.path.join(pathCImg,'Depth','DepthBMP',fileName[0:-4] + '_rotate_'+str(i)+'_'+str(j)+'_'+ax+'_newdepth.bmp'))
                            else:
                                nObj.image.save(nObj.rawRepr[0:-4] + '_rotate_'+str(i)+'_'+str(j)+'_'+ax+'_newdepth.bmp')

                    else:
                        rty = self.getRotationMatrix(i,ax)
                        nObj.image = self.multiplyMatrices(faceCloud,rty)
                        self.outputObj(nObj.image,os.path.join('temporaryTemplate','objRotated',nObj.rawRepr[0:-4] + '_rotate_'+str(i)+'_'+ax+'_newdepth.obj'))
                        nObj = genFaces.doPreProcessing(nObj)
                        nObj.image = im.fromarray(np.array(nObj.image,dtype=np.uint8))
                        nObj.image = nObj.image.rotate(-180)
                        pathCImg = nObj.rawRepr.split(os.path.sep)
                        if pathCImg.index('EURECOM_Kinect_Face_Dataset') >=0 :
                            fileName = pathCImg[-1]
                            pathCImg = os.path.sep.join(pathCImg[0:-2])
                            nObj.image.save(os.path.join(pathCImg,'Depth','DepthBMP',fileName[0:-4] + '_rotate_'+str(i)+'_'+ax+'_newdepth.bmp'))
                        else:
                            nObj.image.save(nObj.rawRepr[0:-4] + '_rotate_'+str(i)+'_'+ax+'_newdepth.bmp')

        return template

if __name__ == '__main__':
    a = [[-37.824600,51.693600,-726.000000],[26.659599,55.858101,-731.000000],[-2.407010,24.070200,-693.000000]]
