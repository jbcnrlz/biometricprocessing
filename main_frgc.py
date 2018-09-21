import sys, argparse
from FRGC import *
from tdlbp import *
#from CenterFace import *
#from SymmetricFilling import *
#from GenerateNewDepthMapsRFRGC import *
#from RotateFaceLFW import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process and extract FRGC database')
    parser.add_argument('-p','--pathdatabase',help='Path for the database',required=True)
    parser.add_argument('-t', '--typeoffile',choices=['Depth', 'NewDepth', 'Range'], help='Type of files (Depth, NewDepth, Range)', required=True)
    parser.add_argument('-op', '--operation',choices=['pp', 'fe', 'both'], default='both', help='Type of operation (pp - PreProcess, fe - Feature Extraction, both)', required=False)
    parser.add_argument('-f', '--pathtrainingfile', default=None,help='Path for the training file', required=False)
    parser.add_argument('-c', '--parcal', default=False,type=bool, help='Should execute in parallell mode?', required=False)
    parser.add_argument('-ap', '--points',type=int,default=None,help='Quantity of points',required=False)
    parser.add_argument('-r', '--radius',type=int,default=None, help='Quantity of points', required=False)
    parser.add_argument('-s', '--steps', default=None, help='Pre-Processing steps, class names separated with _ parameters starts wth : and separated with ,', required=False)

    args = parser.parse_args()

    print('Iniciando...')
    print(args)

    gallery = FRGC(args.pathdatabase,args.typeoffile)
    gallery.feedTemplates()

    tdlbp = ThreeDLBP(8,14,[gallery])
    tdlbp.fullPathGallFile = '/home/joaocardia/PycharmProjects/biometricprocessing/generated_images_lbp_frgc'
    if not args.steps is None:
        ppSteps = args.steps.split('_')
        for p in ppSteps:
            className = None
            parameters = None
            kwargsList = None
            if ':' in p:
                parameters = p.split(':')
                className = parameters[0]
                parameters = parameters[1].split(',')
                kwargsList = {}
                for pr in parameters:
                    lParameters = pr.split('=')
                    kwargsList[lParameters[0]] = eval(lParameters[1])

            else:
                className = p

            module = __import__(className)
            class_ = getattr(module,className)
            if kwargsList is None:
                tdlbp.preProcessingSteps = class_()
            else:
                tdlbp.preProcessingSteps = class_(**kwargsList)

    '''            
    tdlbp.preProcessingSteps = CenterFace()
    tdlbp.preProcessingSteps = SymmetricFilling(regenerate=False)
    tdlbp.preProcessingSteps = RotateFaceLFW()
    tdlbp.preProcessingSteps = GenerateNewDepthMapsRFRGC()
    '''

    if args.operation in ['both','pp']:
        tdlbp.preProcessing(True,args.parcal)

    if args.operation in ['both', 'fe']:
        tdlbp.featureExtraction(args.points,args.radius,args.parcal)

    if not args.pathtrainingfile is None:
        galeryData = gallery.generateDatabaseFile(args.pathtrainingfile)
