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
    
    typeOp - pp, fe or both
    
    exportTrainingFile - Path for the file for training a classifier (SVMTorch model)
    '''

    typeOp = 'both' if len(sys.argv) < 5 else sys.argv[4]
    exportTrainingFile = None if len(sys.argv) < 6 else sys.argv[5]

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

    if typeOp in ['both','pp']:
        tdlbp.preProcessing(True,False)

    if typeOp in ['both', 'fe']:
        tdlbp.featureExtraction()

    if not exportTrainingFile is None:
        galeryData = gallery.generateDatabaseFile(exportTrainingFile)
