import sys
from FRGC import *
from tdlbp import *
from CenterFace import *
from SymmetricFilling import *
from GenerateNewDepthMapsRFRGC import *

if __name__ == '__main__':

    gallery = FRGC('/home/joaocardia/Dropbox/pesquisas/frgc_excerpt/small_frgc/','NewDepth')
    gallery.feedTemplates()
    #gallery.loadNewDepthImage()

    #probe = FRGC('/home/joaocardia/Dropbox/pesquisas/frgc_excerpt/dataset/probe/')
    #probe.feedTemplates()
    #probe.loadNewDepthImage()

    tdlbp = ThreeDLBP(8,14,[gallery])
    tdlbp.preProcessingSteps = CenterFace()
    tdlbp.preProcessingSteps = SymmetricFilling()
    tdlbp.preProcessingSteps = GenerateNewDepthMapsRFRGC()

    #tdlbp.preProcessing(True,False)

    tdlbp.featureExtraction()

    galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/3dlbp_frgc_gallery_noninv_original.txt')
