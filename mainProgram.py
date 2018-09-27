from Eurecom import *
from tdlbp import *
from SegmentFace import *
from TranslateFix import *
from FixWithAveragedModel import *
from FixPaperOcclusion import *
from RotateFace import *
import argparse

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
    parser.add_argument('-gImg', '--pathImages', default='/home/joaocardia/PycharmProjects/biometricprocessing/generated_images_lbp_frgc', help='Path for image signature', required=False)
    parser.add_argument('-v', '--faceVariation',default='Neutral',help='Type of face, separated by _', required=False)
    parser.add_argument('--angles', default=None, help='Angles of face to load',required=False)
    parser.add_argument('--loadNewDepth', default=False, type=bool, help='Load new depth faces', required=False)
    args = parser.parse_args()

    faceDataset = []
    sets = ['s1','s2']
    for s in sets:
        ek = EurecomKinect(args.pathdatabase,s,args.typeoffile,args.faceVariation.split('_'))
        ek.feedTemplates()

        if args.loadNewDepth:
            ek.loadNewDepthImage()

        if args.angles:
            ek.loadRotatedFaces(args.angles.split('_'))

        faceDataset.append(ek)

    tdlbp = ThreeDLBP(8,14,faceDataset)
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

    if args.operation in ['both', 'fe']:
        tdlbp.featureExtraction(args.points,args.radius,args.parcal)

    if not args.pathtrainingfile is None:
        faceVariationGenerate = { k : ['s1','s2'] for k in args.faceVariation.split('_') }
        galeryData = faceDataset[0].generateDatabaseFile(args.pathtrainingfile,faceVariationGenerate,[faceDataset[1]],'SVMTorchFormat')


    #galleryData = probe.generateDatabaseFile('/home/jbcnrlz/Dropbox/pesquisas/BiometricProcessing/generated_images_lbp',{'Neutral' : ['s1','s2']},[],'generateCharsClasses')
    #galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_3dlbp_pr16X1.txt',{
    #    'LightOn' : ['s1','s2'],
    #    'Smile' : ['s1','s2'],
    #    'OpenMouth' : ['s1','s2'],
    #    'OcclusionMouth' : ['s1','s2'],
    #    'OcclusionEyes' : ['s1','s2'],
    #    'Neutral' : ['s1','s2']
    #},[probe],'SVMTorchFormat')
    '''
    galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_3dlbp_oldrotated_102030_full.txt',{
    
    galeryData = gallery.generateDatabaseFile('/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_3dlbp_oldrotated_102030_nonsig.txt',{'OcclusionPaper' : ['s1','s2']},[probe],'SVMTorchFormat')
    probeData = gallery.generateDatabaseFile('/home/jbcnrlz/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/probe_3dlbp_sigmoide.txt',{'Neutral' : ['s1','s2']},[probe],'SVMTorchFormat')
    gallery.applyPCA(52)
    probe.applyPCA(52)
    gallery.generateDatabaseFile('training_set_pca.txt',{'LightOn' : ['s1','s2'],'OcclusionMouth' : ['s1','s2'],'Smile' : ['s1','s2'],'OcclusionEyes' : ['s1','s2']},[probe])
    gallery.generateDatabaseFile('testing_set_pca.txt',{'Neutral' : ['s1','s2']},[probe])
    '''