from Eurecom import *
from WaveletHistogram import *
from FixPose import *
from SegmentFace import *
from SymmetricFilling import *
from TranslateFix import *
from GenerateNewDepthMaps import *
from NormalizaImageDepth import *
from FixWithAveragedModel import *
from FixPaperOcclusion import *
from RotateFace import *
import scipy.ndimage as ndimage, logging, sys

if __name__ == '__main__':
    logging.basicConfig(filename='tdlbp_program_exe.log',
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG)

    logging.info("Testando 3DLBP")

    gallery = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s1','Depth',['LightOn','Neutral','OcclusionMouth','Smile'])
    #gallery = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s1','3DObj',['Neutral'])
    gallery.feedTemplates()
    #gallery.loadSymmFilledImages()

    gallery.loadNewDepthImage()
    #gallery.loadRotatedFaces([10,20,30,-10,-20,-30]) #,,,50,-50

    #gallery.noiseImageGenerate()
    #gallery.loadTemplateImage()
    #gallery.generateAverageFaceModel()

    probe = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s2','Depth',['LightOn','Neutral','OcclusionMouth','Smile'])
    #probe = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s2','3DObj',['Neutral'])
    probe.feedTemplates()
    #probe.loadSymmFilledImages()

    probe.loadNewDepthImage()
    #probe.loadRotatedFaces([10,20,30,-10,-20,-30])#,40,-40,70,-70,80,-80,90,-90,50,-50
    tdlbp = WaveletHistogram(8,80,[probe,gallery],100,100)

    tdlbp.featureExtraction()

    '''
    galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_histogram_wavelet.txt',{
        'Neutral' : ['s1','s2'],
    },[probe],'SVMTorchFormat')
    '''
    
    galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_histogram_wavelet.txt',{
        'LightOn' : ['s1','s2'],
        'Smile' : ['s1','s2'],
        'OpenMouth' : ['s1','s2'],
        'Neutral' : ['s1','s2'],
    },[probe],'SVMTorchFormat')
    