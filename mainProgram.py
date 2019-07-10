from Eurecom import *
from tdlbp import *
import argparse
from helper.functions import sendEmailMessage, standartParametrization

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process and extract FRGC database')
    parser = standartParametrization(parser)
    args = parser.parse_args()

    if not os.path.exists(args.pathImages):
        os.makedirs(args.pathImages)

    faceDataset = []
    sets = ['s1','s2']
    for s in sets:
        ek = EurecomKinect(args.pathdatabase,s,args.typeoffile,args.faceVariation.split('_'))
        ek.feedTemplates()

        if args.loadNewDepth:
            ek.loadNewDepthImage()
        elif args.loadSymmImages:
            ek.loadSymmFilledImages()

        if args.angles:
            ek.loadRotatedFaces(args.angles.split('_'),args.axis.split('_'))

        faceDataset.append(ek)

    tdlbp = ThreeDLBP(8,14,faceDataset,generateImages=args.generateImages)
    tdlbp.fullPathGallFile = args.pathImages

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

    if args.operation in ['both','pp']:
        tdlbp.preProcessing(True,args.parcal)
        sendEmailMessage('Fim do pre-processamento', 'Terminou o pre-processamento EURECOM e LBP')
        #xek.saveTemplateImage()

    if args.operation in ['both', 'fe']:
        tdlbp.featureExtraction(args.points,args.radius,args.parcal,layersUtilize=[1,2,3,4],typeMeasurement=args.typeMeasure,procs=args.quantityProcesses,masks=args.generateMasks,forceImage=args.force,deformValue=args.deformValue)
        sendEmailMessage('Fim dos experimentos','Terminou a extração de características EURECOM e LBP')

    if (args.pathtrainingfile is not None) and (args.operation == 'fe'):
        faceVariationGenerate = { k : ['s1','s2'] for k in args.faceVariation.split('_') }
        galeryData = faceDataset[0].generateDatabaseFile(args.pathtrainingfile,faceVariationGenerate,[faceDataset[1]],'SVMTorchFormat')
