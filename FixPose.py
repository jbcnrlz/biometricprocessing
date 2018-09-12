from baseClasses.PreProcessingStep import *
from helper.functions import getRotationMatrix3D
import math, numpy as np

class FixPose(PreProcessingStep):

    def getMiddlePoint(self,pointA,pointB):
        return [(pointA[0] + pointB[0]) / 2,(pointA[1] + pointB[1]) / 2]

    def findAngle(self,pointA,pointB):
        return math.atan((pointA[0] - pointB[0]) / (pointA[1] - pointB[1]))

    def doPreProcessing(self,template):
        mp = self.getMiddlePoint(template.faceMarks[0][:2],template.faceMarks[1][:2])
        angleRot = self.findAngle(mp,template.faceMarks[2][:2])
        matrixRot = getRotationMatrix3D(angleRot,'z')
        for idx in range(len(template.image)):
            template.image[idx] = matrixRot.dot(np.array(template.image[idx] + [1])).tolist()[:3]

        return template

if __name__ == '__main__':
    from helper.functions import loadOBJ, outputObj, getRotationMatrix, getRotationMatrix3D
    from TranslateFix import *
    from SegmentFace import *
    from SymmetricFilling import *
    from scipy.spatial.distance import euclidean
    import matlab.engine

    sf = SegmentFace()
    tf = TranslateFix()
    we, wq, facepc, y = loadOBJ('/home/jbcnrlz/Documents/eurecom/EURECOM_Kinect_Face_Dataset/0001/s1/3DObj/depth_0001_s1_Neutral.obj')
    facepc = sf.getFaceFromCenterPoint([-2.407010,24.070200,-693.000000],euclidean([-37.824600,51.693600,-726.000000],[26.659599,55.858101,-731.000000]),facepc)
    facepc = tf.translateToOriginByNoseTip(facepc,[-2.407010,24.070200,-693.000000])

    outputObj(facepc,'face_normal.obj')

    a = [-35.41759, 27.623400000000004, -33.0]
    b = [29.066609, 31.787900999999998, -38.0]
    c = [0, 0, 0]

    sm = SymmetricFilling()
    mirroredFace = sm.mirrorFace(facepc)
    faceNP = np.array(facepc)


    outputObj(mirroredFace,'face_mirror.obj')

    eng = matlab.engine.start_matlab()
    tt = eng.doICP()

    tt = np.array(tt)
    
    mirroredList = []
    for idx in range(mirroredFace.shape[0]):
        mirroredList.append(tt.dot(np.array(mirroredFace[idx].tolist() + [1])).tolist()[:3])
    

    outputObj(mirroredList,'lista_espelhada.obj')

    '''
    fp = FixPose()
    mp = fp.getMiddlePoint(a[:2],b[:2])
    angleRot = fp.findAngle(mp,c[:2])
    matrixRot = getRotationMatrix3D(angleRot,'z')
    for idx in range(len(facepcfixed)):
        facepcfixed[idx] = matrixRot.dot(np.array(facepcfixed[idx] + [1])).tolist()[:3]

    outputObj(facepcfixed,'/home/jbcnrlz/Documents/MATLAB/teste_fixed.obj')
    '''



    '''
    facepc = sm.symmetricFillingPCL(np.array(facepc).astype(np.float32),mirroredFace).tolist()

    fp = FixPose()
    mp = fp.getMiddlePoint(a[:2],b[:2])
    angleRot = fp.findAngle(mp,c[:2])
    matrixRot = getRotationMatrix3D(angleRot,'z')
    for idx in range(len(facepc)):
        facepc[idx] = matrixRot.dot(np.array(facepc[idx] + [1])).tolist()[:3]

    outputObj(facepc,'/home/jbcnrlz/Documents/MATLAB/teste_notfixed.obj')
    
    matrixRot = fp.getRotationMatrix(angleRot)
    mpline = matrixRot.dot(mp)
    print(mp)
    print(mpline)
    '''
    

