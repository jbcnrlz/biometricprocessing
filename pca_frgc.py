import argparse
from FRGC import *
from pca import *
from helper.functions import sendEmailMessage, standartParametrization

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process and extract FRGC database')
    parser = standartParametrization(parser)
    args = parser.parse_args()

    print('Iniciando...')
    print(args)

    gallery = FRGC(args.pathdatabase,args.typeoffile)
    gallery.feedTemplates()

    pcaFeature = PCAImpl([gallery],args.pathImages)
    pcaFeature.featureExtraction()