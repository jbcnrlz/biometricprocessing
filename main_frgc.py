import sys, argparse
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

    parser = argparse.ArgumentParser(description='Process and extract FRGC database')
    parser.add_argument('-p','--pathdatabase',help='Path for the database',required=True)
    parser.add_argument('-t', '--typeoffile',choices=['Depth', 'NewDepth', 'Range'], help='Type of files (Depth, NewDepth, Range)', required=True)
    parser.add_argument('-op', '--operation',choices=['pp', 'fe', 'both'], default='both', help='Type of operation (pp - PreProcess, fe - Feature Extraction, both)', required=False)
    parser.add_argument('-ptf', '--pathtrainingfile', default=None,help='Path for the training file', required=False)
    parser.add_argument('-pcall', '--parcal', default=False,type=bool, help='Should execute in parallell mode?', required=False)

    args = parser.parse_args()


    typeOp = 'both' if len(sys.argv) < 4 else sys.argv[3]
    exportTrainingFile = None if len(sys.argv) < 5 else sys.argv[4]

    gallery = FRGC(args.pathdatabase,args.typeoffile)
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

    if args.operation in ['both','pp']:
        tdlbp.preProcessing(True,args.parcal)

    if args.operation in ['both', 'fe']:
        tdlbp.featureExtraction()

    if not args.pathtrainingfile is None:
        galeryData = gallery.generateDatabaseFile(exportTrainingFile)
