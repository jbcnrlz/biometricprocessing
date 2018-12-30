from baseClasses.PreProcessingStep import *
from GenerateNewDepthMapsLFW import *
from GenerateNewDepthMapsRFRGC import *
from FRGCTemplate import *
from PIL import Image as im
import math, numpy as np, copy, pcl

class RotateFaceLFW(PreProcessingStep):

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
            if type(m) == np.ndarray:
                m = m.tolist()
            multMat = np.dot(m+[1],m2)[:-1]
            multiResult.append(multMat)

        return np.array(multiResult)

    def __init__(self,**kwargs):
        self.regenFaces = kwargs.get('regenarate', True)
        self.axis = kwargs.get('axis', ['xy','x','y'])

    def doPreProcessing(self,template):
        genFaces = None
        extentionFiles = 'bmp'
        if type(template) is FRGCTemplate:
            genFaces = GenerateNewDepthMapsRFRGC()
            genFaces.checkForCloud = True
            extentionFiles = 'jpeg'
        else:
            genFaces = GenerateNewDepthMapsLFW()
        faceCloud = None
        if (type(template.image) is not list):
            faceCloud = template.image
        else:
            faceCloud = np.array(template.image)
        #pathFile = template.rawRepr.split(os.path.sep)
        if (not os.path.exists(os.path.join('temporaryTemplate','objRotated'))):
            os.makedirs(os.path.join('temporaryTemplate','objRotated'))
        for ax in self.axis:
            for i in range(-30,40,10):
                if (i != 0):                
                    if (self.regenFaces) or (not os.path.exists(template.rawRepr[0:-4] + '_rotate_'+str(i)+'_'+ax+'_newdepth.'+extentionFiles)):
                        nObj = copy.deepcopy(template)
                        if ax == 'xy':
                            rtx = self.getRotationMatrix(i, 'x')
                            for j in range(-30, 31, 10):
                                if (j == 0):
                                    continue
                                rty = self.getRotationMatrix(j, 'y')
                                nObj.image = self.multiplyMatrices(faceCloud, rtx)
                                nObj.image = self.multiplyMatrices(nObj.image.tolist(), rty)
                                newCloud = pcl.PointCloud(nObj.image.astype(np.float32))
                                fullPathPCD = nObj.rawRepr[0:-4] + '_rotate_' + str(i) + '_' + ax + '.pcd'
                                newCloud.to_file(fullPathPCD.encode('utf-8'))
                                if type(nObj) is FRGCTemplate:
                                    genFaces.fileExtension = '_rotate_' + str(i) + '_' + ax
                                nObj = genFaces.doPreProcessing(nObj)
                                nObj.image = im.fromarray(np.array(nObj.image, dtype=np.uint8))
                                if not type(nObj) is FRGCTemplate:
                                    nObj.image = nObj.image.rotate(-180)
                                    nObj.image.save(
                                        nObj.rawRepr[0:-4] + '_rotate_' + str(i) + '_' + ax + '_newdepth.bmp')

                        else:
                            rty = self.getRotationMatrix(i, ax)
                            nObj.image = self.multiplyMatrices(faceCloud.tolist(), rty)
                            newCloud = pcl.PointCloud(nObj.image.astype(np.float32))
                            fullPathPCD = nObj.rawRepr[0:-4] + '_rotate_' + str(i) + '_' + ax + '.pcd'
                            newCloud.to_file(fullPathPCD.encode('utf-8'))
                            if type(nObj) is FRGCTemplate:
                                genFaces.fileExtension = '_rotate_' + str(i) + '_' + ax
                            nObj = genFaces.doPreProcessing(nObj)
                            nObj.image = im.fromarray(np.array(nObj.image, dtype=np.uint8))
                            if not type(nObj) is FRGCTemplate:
                                nObj.image = nObj.image.rotate(-180)
                                nObj.image.save(nObj.rawRepr[0:-4] + '_rotate_' + str(i) + '_' + ax + '_newdepth.bmp')

        return template
