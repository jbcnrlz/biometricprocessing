import math, numpy as np, pcl, matlab.engine, os, uuid, scipy.ndimage as ndimage
from baseClasses.PreProcessingStep import *
from helper.functions import getArbitraryMatrix, outputObj, loadOBJ
from FRGCTemplate import *

class SymmetricFilling(PreProcessingStep):

    def __init__(self,threshold=0.5,regenerate=True):
        self.regenarate=regenerate
        self.symmThreshold = threshold

    def mirrorFace(self,original):
        mirroredFace = []
        for o in original:
            mirroredFace.append([o[0] * -1,o[1],o[2]])
        return np.array(mirroredFace,dtype=np.float32)

    def applyICP(self,face,mirror,pathCompl=''):
        fileNameUnique = str(uuid.uuid4().hex)
        outputObj(face,os.path.join(pathCompl,fileNameUnique + 'face_normal.obj'))
        outputObj(mirror,os.path.join(pathCompl,fileNameUnique + 'face_mirror.obj'))

        eng = matlab.engine.start_matlab()
        rotMatrice = eng.doICP(pathCompl+os.path.sep+fileNameUnique + 'face_normal.obj',pathCompl+os.path.sep+fileNameUnique + 'face_mirror.obj')
        rotMatrice = np.array(rotMatrice)
        os.remove(os.path.join(pathCompl,fileNameUnique + 'face_normal.obj'))
        os.remove(os.path.join(pathCompl,fileNameUnique + 'face_mirror.obj'))

        mirroredList = []
        for idx in range(mirror.shape[0]):
            mirroredList.append(rotMatrice.dot(np.array(mirror[idx].tolist() + [1])).tolist()[:3])

        return np.array(mirroredList,dtype=np.float32)

    def symmetricFillingPCL(self,face,mirroredFace,pathCompl=''):
        mirroredFace = self.applyICP(face,mirroredFace,pathCompl)
        facet = pcl.PointCloud()
        facet.from_array(face)
        mirror = pcl.PointCloud()
        mirror.from_array(mirroredFace)
        kdeTree = mirror.make_kdtree_flann()
        indices, sqr_distances = kdeTree.nearest_k_search_for_cloud(facet, 2)
        symmetricFilledFace = []
        for x in range(mirror.size):
            if (sqr_distances[x][1] > self.symmThreshold):
                symmetricFilledFace.append(mirror[x])
        symfacecon = np.array(symmetricFilledFace,dtype=np.float32)
        return np.concatenate((face,symfacecon))

    def smoothCloudPoint(self,cp):
        facet = pcl.PointCloud()
        facet.from_array(cp)
        fil = facet.make_statistical_outlier_filter()
        fil.set_mean_k (50)
        fil.set_std_dev_mul_thresh (1.0)
        return np.array(fil.filter(),dtype=np.float32)

    def outputSymmFillObj(self,template):
        subject = "%04d" % (template.itemClass)
        template3Dobj = template.rawRepr.split(os.path.sep)[:-3]
        folderType = template3Dobj[template3Dobj.index(subject) + 1]
        outputObj(template.image,os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'_symmetricfilled.obj'))

    def doPreProcessing(self,template):
        if (not self.regenarate) and (os.path.exists(template.rawRepr[:-4] + '_symfilled.obj')):
            a, b, template.image, d = loadOBJ(template.rawRepr[:-4] + '_symfilled.obj')
            return template
        imageFromFace = np.array(template.image,dtype=np.float32)
        mirroredFace = self.mirrorFace(imageFromFace)
        folderPath = template.rawRepr.split(os.path.sep)
        faceSimmetricalFused = self.symmetricFillingPCL(imageFromFace,mirroredFace,self.symmThreshold,os.path.sep.join(folderPath[:-1]))
        if not type(template) is FRGCTemplate:
            template.image = faceSimmetricalFused.tolist()
            self.outputSymmFillObj(template)
        else:
            template.save(True, '_symfilled')
        return template
