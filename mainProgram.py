from Eurecom import *
from tdlbp import *
from FixPose import *
from SegmentFace import *
from SymmetricFilling import *
from TranslateFix import *
from GenerateNewDepthMaps import *
from NormalizaImageDepth import *
from FixWithAveragedModel import *
from FixPaperOcclusion import *
from RotateFace import *
#from helper.lmdbGeneration import load_data_into_lmdb
import logging, sys, argparse
#import caffe

if __name__ == '__main__':

    gallery = EurecomKinect('/home/joaocardia/PycharmProjects/EURECOM_Kinect_Face_Dataset','s1','Depth',['LightOn','Neutral','Smile','OpenMouth'])
    #gallery = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s1','Depth',['OcclusionPaper'])
    gallery.feedTemplates()

    gallery.loadNewDepthImage()
    #gallery.loadRotatedFaces([10,20,30,-10,-20,-30]) #,,,50,-50

    #gallery.noiseImageGenerate()
    #gallery.loadTemplateImage()
    #gallery.generateAverageFaceModel()

    probe = EurecomKinect('/home/joaocardia/PycharmProjects/EURECOM_Kinect_Face_Dataset','s2','Depth',['LightOn','Neutral','Smile','OpenMouth'])
    #probe = EurecomKinect('/home/joaocardia/Dropbox/Mestrado CC/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset','s2','Depth',['OcclusionPaper'])
    probe.feedTemplates()

    probe.loadNewDepthImage()
    #probe.loadRotatedFaces([10,20,30,-10,-20,-30])#,40,-40,70,-70,80,-80,90,-90,50,-50
    
    #probe.noiseImageGenerate()
    #probe.loadTemplateImage()
    #probe.fixingProbe()

    tdlbp = ThreeDLBP(8,14,[probe,gallery])
    tdlbp.preProcessingSteps = SegmentFace()
    tdlbp.preProcessingSteps = TranslateFix()
    tdlbp.preProcessingSteps = SymmetricFilling()    
    tdlbp.preProcessingSteps = FixPaperOcclusion()
    tdlbp.preProcessingSteps = RotateFace()
    tdlbp.preProcessingSteps = GenerateNewDepthMaps()

    #tdlbp.preProcessingSteps = NormalizaImageDepth()    

    #tdlbp.preProcessing(True)

    #gallery.saveTemplateImage()
    #probe.saveTemplateImage()    

    tdlbp.fullPathGallFile = '/home/joaocardia/PycharmProjects/biometricprocessing/generated_images_eurecom'
    tdlbp.featureExtraction()
    
    #resultados = tdlbp.matcher()
    
    #print(resultados)
    
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