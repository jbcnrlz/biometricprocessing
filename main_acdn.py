from Eurecom import *
from acdn import *
from FixPose import *
from SegmentFace import *
from SymmetricFilling import *
from TranslateFix import *
from SmoothImage import *
from NormalizaImageDepth import *
from FixWithAveragedModel import *
from FixPaperOcclusion import *
import scipy.ndimage as ndimage

if __name__ == '__main__':

    gallery = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s1','3DObj',['LightOn','Neutral','OcclusionMouth','Smile','OcclusionEyes','OpenMouth','OcclusionPaper'])
    gallery.feedTemplates(True)
    gallery.loadSymmFilledImages()
    gallery.loadRotatedFacesPC([10,20,30,-10,-20,-30]) #,,,50,-50

    probe = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s2','3DObj',['LightOn','Neutral','OcclusionMouth','Smile','OcclusionEyes','OpenMouth','OcclusionPaper'])
    probe.feedTemplates(True)
    probe.loadSymmFilledImages()
    probe.loadRotatedFacesPC([10,20,30,-10,-20,-30]) #,,,50,-50

    ac = acdn([gallery,probe])
    ac.preProcessingSteps = SegmentFace()
    ac.preProcessingSteps = TranslateFix()
    ac.preProcessingSteps = SymmetricFilling()    
    ac.preProcessingSteps = FixPaperOcclusion()
    #tdlbp.preProcessingSteps = SmoothImage()

    #ac.preProcessing(True)

    ac.featureExtraction()

    galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/base_acdnp_facehidden.txt',{
        'LightOn' : ['s1','s2'],
        'Smile' : ['s1','s2'],
        'OpenMouth' : ['s1','s2'],
        'OcclusionMouth' : ['s1','s2'],
        'OcclusionEyes' : ['s1','s2'],
        'Neutral' : ['s1','s2'],
        'OcclusionPaper' : ['s1','s2']
    },[probe],'SVMTorchFormat')
    '''
    probeData = gallery.generateDatabaseFile('/home/jbcnrlz/Dropbox/pesquisas/BiometricProcessing/generated_images_lbp/test_acdn.txt',{
        'Neutral' : ['s1','s2']
    },[probe],'SVMTorchFormat')
    '''
