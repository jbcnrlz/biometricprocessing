from Eurecom import *
from wld import *
from FixPose import *
from SegmentFace import *
from SymmetricFilling import *
from TranslateFix import *
from GenerateNewDepthMaps import *
from NormalizaImageDepth import *
from FixWithAveragedModel import *
from RotateFace import *
#from helper.lmdbGeneration import load_data_into_lmdb
import scipy.ndimage as ndimage, logging, sys
#import caffe

if __name__ == '__main__':
    logging.basicConfig(filename='wld_program_exe.log',
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG)

    logging.info("Testando WLD")

    gallery = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s1','Depth',['LightOn','Neutral','OcclusionMouth','Smile','OcclusionEyes','OpenMouth'])
    gallery.feedTemplates()
    gallery.loadNewDepthImage()
    gallery.loadRotatedFaces([10,20,30,-10,-20,-30]) #,,,50,-50

    probe = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s2','Depth',['LightOn','Neutral','OcclusionMouth','Smile','OcclusionEyes','OpenMouth'])
    probe.feedTemplates()
    probe.loadNewDepthImage()
    probe.loadRotatedFaces([10,20,30,-10,-20,-30])#,40,-40,70,-70,80,-80,90,-90,50,-50

    wld = WLD(15,8,[probe,gallery])
    wld.featureExtraction()

    galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_3dlbp_oldrotated_102030_full.txt',{
        'LightOn' : ['s1','s2'],
        'Smile' : ['s1','s2'],
        'OpenMouth' : ['s1','s2'],
        'OcclusionMouth' : ['s1','s2'],
        'OcclusionEyes' : ['s1','s2'],
        'Neutral' : ['s1','s2']
    },[probe],'SVMTorchFormat')