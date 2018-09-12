from Eurecom import *
from CloudLBP import *
from FixPose import *
from SegmentFace import *
from SymmetricFilling import *
from TranslateFix import *
from SmoothImage import *
from NormalizaImageDepth import *
from FixWithAveragedModel import *
import scipy.ndimage as ndimage

if __name__ == '__main__':

    gallery = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s1','3DObj',['Neutral'])
    gallery.feedTemplates()
    gallery.loadSymmFilledImages()

    probe = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s2','3DObj',['Neutral'])
    probe.feedTemplates()
    probe.loadSymmFilledImages()

    tdlbp = CloudLBP(7,14,[gallery,probe])

    tdlbp.featureExtraction()

    galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/data_cloudlbp.txt',{
        'Neutral' : ['s1','s2']
    },[probe],'SVMTorchFormat')