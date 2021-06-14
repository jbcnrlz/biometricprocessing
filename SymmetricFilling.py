import uuid
from baseClasses.PreProcessingStep import *
from helper.functions import getArbitraryMatrix, outputObj, loadOBJ, icp, nearest_neighbor
from FRGCTemplate import *
from IIITDTemplate import *
from matlab import engine

class SymmetricFilling(PreProcessingStep):

    def __init__(self,**kwargs):
        self.regenarate=kwargs.get('regenarate',True)
        self.symmThreshold = kwargs.get('symmThreshold',0.5)

    def mirrorFace(self,original):
        mirroredFace = []
        for o in original:
            mirroredFace.append([o[0] * -1,o[1],o[2]])
        return np.array(mirroredFace,dtype=np.float32)

    def applyICP(self,face,mirror,pathCompl=''):
        fileNameUnique = str(uuid.uuid4().hex)
        outputObj(face,os.path.join(pathCompl,fileNameUnique + 'face_normal.obj'))
        outputObj(mirror,os.path.join(pathCompl,fileNameUnique + 'face_mirror.obj'))

        eng = engine.start_matlab()
        rotMatrice = eng.doICP(pathCompl+os.path.sep+fileNameUnique + 'face_normal.obj',pathCompl+os.path.sep+fileNameUnique + 'face_mirror.obj')
        rotMatrice = np.array(rotMatrice)
        os.remove(os.path.join(pathCompl,fileNameUnique + 'face_normal.obj'))
        os.remove(os.path.join(pathCompl,fileNameUnique + 'face_mirror.obj'))

        mirroredList = []
        for idx in range(mirror.shape[0]):
            mirroredList.append(rotMatrice.dot(np.array(mirror[idx].tolist() + [1])).tolist()[:3])

        return np.array(mirroredList,dtype=np.float32)

    '''
    def symmetricFillingPCL(self,face,mirroredFace,pathCompl='',doICP=True):
        if doICP:
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
    '''

    def symmetricFilling(self,face,mirroredFace,pathCompl='',doICP=True):
        if doICP:
            t,d,i = icp(mirroredFace,face)

        distances, indices = nearest_neighbor(face, mirroredFace)
        symmetricFilledFace = []
        for i, x in enumerate(indices):
            if (distances[i] > self.symmThreshold):
                    symmetricFilledFace.append(mirroredFace[x])

        if len(symmetricFilledFace) == 0:
            return face
        symfacecon = np.array(symmetricFilledFace,dtype=np.float32)
        coordnates = np.ones((symfacecon.shape[0], 1))
        sface = np.concatenate((symfacecon, coordnates), axis=1)
        symfacecon = t.dot(sface.T).T
        symfacecon = symfacecon[:,:3]
        return np.concatenate((face,symfacecon))

    def outputSymmFillObj(self,template):
        subject = "%04d" % (template.itemClass)
        template3Dobj = template.rawRepr.split(os.path.sep)[:-2]
        folderType = template3Dobj[template3Dobj.index(subject) + 1]
        outputObj(template.image,os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'_symmetricfilled.obj'))

    def doPreProcessing(self,template):
        if (not self.regenarate) and (os.path.exists(template.rawRepr[:-4] + '_symfilled.obj')):
            a, b, template.image, d = loadOBJ(template.rawRepr[:-4] + '_symfilled.obj')
            return template
        imageFromFace = np.array(template.image,dtype=np.float32)
        mirroredFace = self.mirrorFace(imageFromFace)
        folderPath = template.rawRepr.split(os.path.sep)
        faceSimmetricalFused = self.symmetricFilling(imageFromFace,mirroredFace,os.path.sep.join(folderPath[:-1]),doICP=(template.typeTemplate != 'OcclusionPaper'))
        if type(template) is IIITDTemplate:
            template.image = faceSimmetricalFused.tolist()
            template.saveSymfilled()
        elif not type(template) is FRGCTemplate:
            template.image = faceSimmetricalFused.tolist()
            self.outputSymmFillObj(template)
        else:
            template.save(True, '_symfilled')
        return template
