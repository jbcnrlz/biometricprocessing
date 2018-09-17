import sys
from FRGC import *
from tdlbp import *
from CenterFace import *
from SymmetricFilling import *
from GenerateNewDepthMapsRFRGC import *
from RotateFaceLFW import *

if __name__ == '__main__':

    '''
    How to use:
    python main_frgc.py pathDatabase typeOfFile [typeOp] [exportTrainingFile]
    
    pathDatabase - Path of the Database (FRGC)
    
    typeOfFile - Depth, NewDepth, Range
    
    typeOp - Pre-Process, Feature Extraction or Both
    
    exportTrainingFile - Path for the file for training a classifier (SVMTorch model)
    '''

    gallery = FRGC(sys.argv[1],sys.argv[2])
    gallery.feedTemplates()
    #gallery.loadNewDepthImage()

    #probe = FRGC('/home/joaocardia/Dropbox/pesquisas/frgc_excerpt/dataset/probe/')
    #probe.feedTemplates()
    #probe.loadNewDepthImage()

    tdlbp = ThreeDLBP(8,14,[gallery])
    tdlbp.preProcessingSteps = CenterFace()
    tdlbp.preProcessingSteps = SymmetricFilling()
    tdlbp.preProcessingSteps = RotateFaceLFW()
    tdlbp.preProcessingSteps = GenerateNewDepthMapsRFRGC()

    #tdlbp.preProcessing(True,False)

    tdlbp.featureExtraction()

    galeryData = gallery.generateDatabaseFile(sys.argv[3])
