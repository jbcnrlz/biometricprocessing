from baseClasses.PreProcessingStep import *
from GenerateNewDepthMaps import *
from PIL import Image as im
from removeSelfOcclusion import estimateVis_vertex
import math, open3d as o3d, numpy as np, copy

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

    def localGetRotationMatrix(self,angle,matrix):
        cosResult = np.cos(np.radians(angle))
        sinResult = np.sin(np.radians(angle))
        if matrix == 'x':
            return np.array([
                (1 ,0, 0),
                (0 , cosResult, -sinResult),
                (0 , sinResult,  cosResult)
            ])
        elif matrix == 'y':
            return np.array([
                (cosResult , 0, sinResult),
                (0         , 1, 0        ),
                (-sinResult, 0, cosResult),
            ])
        else:
            return np.array([
                (cosResult ,-sinResult, 0),
                (sinResult , cosResult, 0),
                (0         ,         0, 1),
            ])

    def doPreProcessing(self,template):
        genFaces = GenerateNewDepthMaps()
        faceCloud = template.image
        if type(faceCloud) is list:
            faceCloud = np.array(faceCloud)
        #pathFile = template.rawRepr.split(os.path.sep)
        if (not os.path.exists(os.path.join('temporaryTemplate','objRotated'))):
            os.makedirs(os.path.join('temporaryTemplate','objRotated'))

        for ax in self.axis:
            for i in range(-90,91,10):
                if (i != 0):
                    nObj = copy.deepcopy(template)
                    if ax == 'xy':
                        rtx = self.getRotationMatrix(i, 'x')
                        for j in range(-30,31,10):
                            rty = self.getRotationMatrix(j, 'y')
                            nObj.image = self.multiplyMatrices(faceCloud, rtx)
                            nObj.image = self.multiplyMatrices(nObj.image.tolist(), rty)

                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(nObj.image)
                            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                                     std_ratio=0.5)
                            inlier_cloud = pcd.select_down_sample(ind)
                            nObj.image = np.asarray(inlier_cloud.points)


                            self.outputObj(nObj.image,os.path.join('temporaryTemplate','objRotated',nObj.rawRepr[0:-4] + '_rotate_'+str(i)+'_'+str(j)+'_'+ax+'_newdepth.obj'))
                            nObj = genFaces.doPreProcessing(nObj)
                            nObj.image = im.fromarray(np.array(nObj.image,dtype=np.uint8))
                            nObj.image = nObj.image.rotate(-180)
                            pathCImg = nObj.rawRepr.split(os.path.sep)
                            if pathCImg.index('EURECOM_Kinect_Face_Dataset') >=0 :
                                fileName = pathCImg[-1].split('_')
                                pathCImg = os.path.sep.join(pathCImg[0:-2])
                                nObj.image.save(os.path.join(pathCImg,'Depth','DepthBMP','_'.join(fileName[0:4]) + '_rotate_'+str(i)+'_'+ax+'_newdepth.bmp'))
                            else:
                                nObj.image.save(nObj.rawRepr[0:-4] + '_rotate_'+str(i)+'_'+str(j)+'_'+ax+'_newdepth.bmp')

                    else:
                        rty = self.getRotationMatrix(i,ax)
                        if faceCloud.shape[1] < 4:
                            uns = np.ones((faceCloud.shape[0],1))
                            faceCloud = np.concatenate((faceCloud, uns), axis=1)
                        nObj.image = self.multiplyMatrices(faceCloud,rty)
                        #rmvp = self.localGetRotationMatrix(i,ax)
                        #visparts = estimateVis_vertex(nObj.image,rmvp,300,4)
                        #visparts = visparts[visparts >=0]
                        #nObj.image = nObj.image[visparts]

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(nObj.image)
                        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
                        camera = [0, 0, diameter]
                        radius = diameter * 100

                        _, pt_map = pcd.hidden_point_removal(camera, radius)
                        #pcd = pcd.select_by_index(pt_map)

                        nObj.image = np.asarray(pcd.points)[pt_map]

                        self.outputObj(nObj.image,os.path.join('temporaryTemplate','objRotated',nObj.rawRepr[0:-4] + '_rotate_'+str(i)+'_'+ax+'_newdepth.obj'))
                        nObj = genFaces.doPreProcessing(nObj)
                        nObj.image = im.fromarray(np.array(nObj.image,dtype=np.uint8))
                        nObj.image = nObj.image.rotate(-180)
                        pathCImg = nObj.rawRepr.split(os.path.sep)
                        if pathCImg.index('EURECOM_Kinect_Face_Dataset') >=0 :
                            fileName = pathCImg[-1].split('_')
                            pathCImg = os.path.sep.join(pathCImg[0:-2])
                            nObj.image.save(os.path.join(pathCImg,'Depth','DepthBMP','_'.join(fileName[0:4]) + '_rotate_'+str(i)+'_'+ax+'_newdepth.bmp'))
                        else:
                            nObj.image.save(nObj.rawRepr[0:-4] + '_rotate_'+str(i)+'_'+ax+'_newdepth.bmp')

        return template

if __name__ == '__main__':
    a = [[-37.824600,51.693600,-726.000000],[26.659599,55.858101,-731.000000],[-2.407010,24.070200,-693.000000]]
