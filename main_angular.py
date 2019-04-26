import argparse
from FRGC import *
from angularFeatureExtraction import AngularFeatureExtraction
from helper.functions import sendEmailMessage, standartParametrization

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process and extract FRGC database')
    parser = standartParametrization(parser)
    args = parser.parse_args()

    print('Iniciando...')
    print(args)

    gallery = FRGC(args.pathdatabase,args.typeoffile)
    gallery.feedTemplates()

    ang = AngularFeatureExtraction([gallery])
    ang.fullPathGallFile = args.pathImages

    if args.operation in ['both', 'fe']:
        ang.featureExtraction(
            args.points,
            args.radius,
            args.parcal,
            procs=args.quantityProcesses,
            forceImage=args.force,
        )
        sendEmailMessage('Fim dos experimentos', 'Terminou a extração de características FRGC e LBP')

