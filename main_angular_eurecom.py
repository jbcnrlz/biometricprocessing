import argparse
from Eurecom import *
from angularFeatureExtraction import AngularFeatureExtraction
from helper.functions import sendEmailMessage, standartParametrization

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process and extract EURECOM database')
    parser = standartParametrization(parser)
    args = parser.parse_args()

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

    ang = AngularFeatureExtraction(faceDataset)
    ang.fullPathGallFile = args.pathImages

    if args.operation in ['both', 'fe']:
        ang.featureExtraction(
            args.points,
            args.radius,
            args.parcal,
            procs=args.quantityProcesses,
            forceImage=args.force,
        )
        sendEmailMessage('Fim dos experimentos','Terminou a extração de características EURECOM e Angular')