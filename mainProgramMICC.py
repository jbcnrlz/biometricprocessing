import argparse
from MICCDataset import *
from tdlbp import *
from helper.functions import sendEmailMessage, standartParametrization

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process and extract MICC database')
    parser = standartParametrization(parser)
    args = parser.parse_args()

    print('Iniciando...')
    print(args)

    gallery = MICCDataset(args.pathdatabase,args.typeoffile)
    gallery.feedTemplates()

    if args.loadNewDepth:
        gallery.loadNewDepthImage()

    if args.angles:
        gallery.loadRotatedFaces(args.angles.split('_'),args.axis.split('_'))

    tdlbp = ThreeDLBP(8,14,[gallery])
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
        sendEmailMessage('Fim do pre-processamento', 'Terminou o pre-processamento FRGC e LBP')

    if args.operation in ['both', 'fe']:
        if not os.path.exists(args.pathImages):
            os.makedirs(args.pathImages)

        tdlbp.featureExtraction(
            args.points,
            args.radius,
            args.parcal,
            procs=args.quantityProcesses,
            masks=args.generateMasks,
            forceImage=args.force,
            typeMeasurement=args.typeMeasure,
            firstLayer=args.firstLayer
        )
        sendEmailMessage('Fim dos experimentos', 'Terminou a extração de características FRGC e LBP')

    if not args.pathtrainingfile is None:
        galeryData = gallery.generateDatabaseFile(args.pathtrainingfile)
