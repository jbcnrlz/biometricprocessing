from Eurecom import *
from acdn import *
from tdlbp import *
from FixPose import *
from SegmentFace import *
from SymmetricFilling import *
from TranslateFix import *
from SmoothImage import *
from NormalizaImageDepth import *
from FixWithAveragedModel import *
from GenerateNewDepthMaps import *
import scipy.ndimage as ndimage
from baseClasses.MethodFusion import *

if __name__ == '__main__':

    galleryACDN = EurecomKinect('/home/jbcnrlz/Documents/eurecom/EURECOM_Kinect_Face_Dataset','s1','Depth',['Neutral'])
    galleryACDN.feedTemplates()

    probeACDN = EurecomKinect('/home/jbcnrlz/Documents/eurecom/EURECOM_Kinect_Face_Dataset','s2','Depth',['Neutral'])
    probeACDN.feedTemplates()
    ac = acdn([galleryACDN,probeACDN])
    ac.preProcessingSteps = SegmentFace()
    ac.preProcessingSteps = TranslateFix()
    ac.preProcessingSteps = SymmetricFilling()
    ac.preProcessing(True)
    ac.featureExtraction()
    print(ac.matcher(True))
    
    galleryTDLBP = EurecomKinect('/home/jbcnrlz/Documents/eurecom/small_eurecom','s1','Depth',['Neutral'])
    galleryTDLBP.feedTemplates()
    probeTDLBP = EurecomKinect('/home/jbcnrlz/Documents/eurecom/small_eurecom','s2','Depth',['Neutral'])
    probeTDLBP.feedTemplates()    
    tdlbp = ThreeDLBP(3,14,[galleryTDLBP,probeTDLBP])
    tdlbp.preProcessingSteps = SegmentFace()
    tdlbp.preProcessingSteps = TranslateFix()
    tdlbp.preProcessingSteps = SymmetricFilling()
    tdlbp.preProcessingSteps = GenerateNewDepthMaps()
    tdlbp.preProcessingSteps = NormalizaImageDepth()    
    tdlbp.preProcessing(True)
    tdlbp.featureExtraction()
    print(tdlbp.matcher())

    mf = MethodFusion()
    mf.galleries = galleryACDN
    mf.galleries = galleryTDLBP
    mf.probes = probeACDN
    mf.probes = probeTDLBP
    mf.doFusion()
    mf.matcher()
