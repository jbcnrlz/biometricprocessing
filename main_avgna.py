from Eurecom import *
from AVGNormalAngles import *
from FixPose import *
from GenerateNewDepthMaps3Channels import *
import scipy.ndimage as ndimage

if __name__ == '__main__':

    gallery = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s1','3DObj',['Neutral'])
    gallery.feedTemplates()
    gallery.loadSymmFilledImages()
    gallery.loadRotatedFacesPC([10,20,30,-10,-20,-30]) #,,,50,-50

    probe = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s2','3DObj',['Neutral'])
    probe.feedTemplates()
    probe.loadSymmFilledImages()
    probe.loadRotatedFacesPC([10,20,30,-10,-20,-30]) #,,,50,-50

    ac = AVGNormalAngles((20,20),14,[gallery,probe])
    #ac.preProcessingSteps = GenerateNewDepthMaps3Channels()
    #ac.preProcessingSteps = TranslateFix()
    #ac.preProcessingSteps = SymmetricFilling()
    #tdlbp.preProcessingSteps = SmoothImage()

    #ac.preProcessing(True)

    ac.featureExtraction()
    #gallery.normalizeData()
    #probe.normalizeData()
    '''
    galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/base_avgna_training_rotate.txt',{
        'LightOn' : ['s1','s2'],
        'Smile' : ['s1','s2'],
        'OpenMouth' : ['s1','s2'],
        'OcclusionMouth' : ['s1','s2'],
        'OcclusionEyes' : ['s1','s2'],
        'Neutral' : ['s1','s2']
    },[probe],'SVMTorchFormat')
    '''
    probeData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/base_avgna_training_rotate_bin14Img2020.txt',{
        'Neutral' : ['s1','s2']
    },[probe],'SVMTorchFormat')