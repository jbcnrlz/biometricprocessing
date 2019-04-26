from Eurecom import *
from pca                     import *
import argparse
from helper.functions import standartParametrization

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process and extract EURECOM with PCA')
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

    pcaFeature = PCAImpl(faceDataset,args.pathImages)
    pcaFeature.featureExtraction()