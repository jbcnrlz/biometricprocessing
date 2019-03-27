import argparse
from BU3DFE import *
from tdlbp import *
from helper.functions import standartParametrization, sendEmailMessage

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process and extract BU3DFE database')
    parser = standartParametrization(parser)
    args = parser.parse_args()

    print('Iniciando...')
    print(args)

    gallery = BU3DFE(args.pathdatabase,args.typeoffile,args.faceVariation)
    gallery.generateTranslationTab()
    if args.loadImages is None:
        gallery.feedTemplates()
    else:
        gallery.feedTemplatesFromList(args.loadImages.split('__'))

    if args.loadNewDepth:
        gallery.loadNewDepthImage()

    if args.angles:
        gallery.loadRotatedFaces(args.angles.split('_'),['x','y'])

    tdlbp = ThreeDLBP(8,14,[gallery])
    tdlbp.fullPathGallFile = args.pathImages

    if args.operation in ['both','pp']:
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
                        kwargsList[lParameters[0]] = lParameters[1]

                else:
                    className = p

                module = __import__(className)
                class_ = getattr(module, className)
                if kwargsList is None:
                    tdlbp.preProcessingSteps = class_()
                else:
                    tdlbp.preProcessingSteps = class_(**kwargsList)

        tdlbp.preProcessing(True,args.parcal)
        #gallery.saveNewDepthImages()
        sendEmailMessage('Fim do pre-processamento', 'Terminou o pre-processamento BU e ' + args.typeMeasure)

    if args.operation in ['both', 'fe']:
        tdlbp.featureExtraction(args.points,args.radius,args.parcal,forceImage=args.force,typeMeasurement=args.typeMeasure,procs=args.quantityProcesses)
        sendEmailMessage('Fim do pre-processamento', 'Terminou a extração de características BU e ' + args.typeMeasure)

    if not args.pathtrainingfile is None:
        galeryData = gallery.generateDatabaseFile(args.pathtrainingfile)
