from math import sqrt
from baseClasses.Template import *
import os, numpy as np, random
from helper.functions import outputObj, loadOBJ, scaleValues
from PIL import Image as im

class EurecomTemplate(Template):

    folderTemplate = None
    faceMarks = []
    layersChar = None
    overFlow = None
    underFlow = None

    def loadImage(self):
        if self.rawRepr[-3:] == 'bmp':
            imageFace = im.open(self.rawRepr).convert('L')
            self.layersChar = np.zeros((imageFace.size[0],imageFace.size[1],4))
        else:
            a, b, imageFace, y = loadOBJ(os.path.join('temporaryTemplate',str(self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.obj'))
        self.image = imageFace

    def loadSymFilledImage(self):
        if self.lazyLoading:
            self.rawRepr = self.rawRepr[0:-4] + '_symmetricfilled.obj'
        else:
            a, b, imageFace, y = loadOBJ(self.rawRepr[0:-4] + '_symmetricfilled.obj')
            self.image = imageFace

    def save(self,saveOnPath=False):
        if (not saveOnPath):
            if (not os.path.exists('temporaryTemplate')):
                os.makedirs('temporaryTemplate')

            outputObj(self.image,os.path.join('temporaryTemplate',str(self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.obj'))
            self.outputMarks()
        else:
            pathCImg = self.rawRepr.split(os.path.sep)
            if pathCImg.index('EURECOM_Kinect_Face_Dataset') >= 0:
                fileName = pathCImg[-1]
                pathCImg = os.path.sep.join(pathCImg[0:-2])
                self.image.save(os.path.join(pathCImg, 'Depth', 'DepthBMP',
                                             fileName[0:-4] + '_newdepth.bmp'))
            else:
                self.image.save(self.rawRepr[0:-4] + '_newdepth.bmp')

    def existsPreProcessingFile(self):
        fullImgPath = ''
        pathCImg = self.rawRepr.split(os.path.sep)
        if pathCImg.index('EURECOM_Kinect_Face_Dataset') >= 0:
            fileName = pathCImg[-1]
            pathCImg = os.path.sep.join(pathCImg[0:-2])
            fullImgPath = os.path.join(pathCImg, 'Depth', 'DepthBMP',
                                         fileName[0:-4] + '_newdepth.bmp')
        else:
            fullImgPath = self.rawRepr[0:-4] + '_newdepth.bmp'

        return os.path.exists(fullImgPath)

    def outputMarks(self,saveOnPath=False,typeTemplate='Depth'):
        if (not saveOnPath):
            if (not os.path.exists('temporaryTemplate')):
                os.makedirs('temporaryTemplate')
            f = open(os.path.join('temporaryTemplate',str(self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.txt'),'w')
            f.write('\n'.join([ '\t'.join(map(str,x)) for x in self.faceMarks]))
            f.close()
        else:
            filesPath = self.rawRepr.split('/')
            fileName = filesPath[-1].split('.')
            fileName = fileName[0].split('_')
            if typeTemplate == 'Depth':
                filesPath = os.path.join('/'.join(filesPath[:-3]),'Mark','MarkRGB','rgb_'+fileName[1]+'_'+filesPath[:-3][len(filesPath) - 4]+'_'+fileName[3]+'_Points_newdepth.txt')
            elif typeTemplate == '3DObj':
                filesPath = os.path.join('/'.join(filesPath[:-3]),'Mark','Mark3DObj','depth_'+fileName[1]+'_'+filesPath[:-3][len(filesPath) - 4]+'_'+fileName[3]+'_Points_OBJ_newdepth.txt')
            f = open(filesPath,'w')
            f.write('\n'.join([ '\t'.join(map(str,x)) for x in self.faceMarks]))
            f.close()

    def loadMarks(self,typeTemplate='Depth'):
        if os.path.sep in typeTemplate:
            typeTemplate = typeTemplate.split(os.path.sep)
            typeTemplate = typeTemplate[0]
        filesPath = self.rawRepr.split(os.path.sep)
        fileName = filesPath[-1].split('.')
        fileName = fileName[0].split('_')
        if typeTemplate.lower() == 'depth':
            filesPath = os.path.join(os.path.sep.join(filesPath[:-3]),'Mark','MarkRGB','rgb_'+fileName[1]+'_'+filesPath[:-3][len(filesPath) - 4]+'_'+fileName[3]+'_Points.txt')
        elif typeTemplate.lower() == '3dobj':
            filesPath = os.path.join(os.path.sep.join(filesPath[:-3]),'Mark','Mark3DObj','depth_'+fileName[1]+'_'+filesPath[:-3][len(filesPath) - 4]+'_'+fileName[3]+'_Points_OBJ.txt')
        elif typeTemplate.lower() == 'newdepth':
            filesPath = os.path.join(os.path.sep.join(filesPath[:-3]),'Mark','MarkRGB','rgb_'+fileName[1]+'_'+filesPath[:-3][len(filesPath) - 4]+'_'+fileName[3]+'_Points_newdepth.txt')
        self.faceMarks = []
        if (os.path.exists(filesPath)):
            fileMark = open(filesPath,'r')
            for p in fileMark:
                self.faceMarks.append(list(map(float,p.split('\t'))))
        else:
            filesPath = self.rawRepr.split(os.path.sep)
            filesPath = os.path.join(os.path.sep.join(filesPath[:-2]),'Mark','Mark3DObj','depth_'+fileName[1]+'_'+self.folderTemplate+'_'+fileName[3]+'_Points_OBJ.txt')
            fileMark = open(filesPath,'r')
            for p in fileMark:
                self.faceMarks.append(list(map(float,p.split('\t'))))


    def saveTXTChars(self):
        f = open('teste.txt','w')
        f.write(' '.join(map(str,self.features)) + '\n')
        f.close()

    def loadNewDepthImage(self):
        self.image = im.open(self.rawRepr[0:-4] + '_newdepth.bmp')
        self.loadMarks('newdepth')

    def saveImageTraining(self,avgImageSave=True,pathImage='generated_images_lbp'):
        if (avgImageSave):
            avImage = np.zeros((self.layersChar.shape[0],self.layersChar.shape[1]))        
            for i in range(self.layersChar.shape[0]):
                for j in range(self.layersChar.shape[1]):
                    avImage[i,j] = self.layersChar[i,j,0] + self.layersChar[i,j,1] + self.layersChar[i,j,2] + self.layersChar[i,j,3]
                    avImage[i,j] = avImage[i,j] / 4
            avImage = im.fromarray(np.uint8(avImage))
        
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        self.layersChar = scaleValues(0,255,self.layersChar)
        imageSaveDLP = im.fromarray(np.uint8(self.layersChar))
        pathNImage = pathImage+'/'+str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath +'.png'
        imageSaveDLP.save(pathNImage)

    def saveHistogramImage(self,imageSave=None,folder='generated_images_wld'):
        if (imageSave is None):
            imageSave = self.features

        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = im.fromarray(imageSave)
        pathNImage = folder + '/'+str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath +'.jpg'
        while (os.path.exists(pathNImage)):
            idxRandomIm = random.randint(1,255)
            pathNImage = folder+'/'+str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath +'_'+str(idxRandomIm)+'.png'

        imageSaveDLP.convert('RGB').save(pathNImage)

    def saveMasks(self,folder,filetype):
        if self.overFlow is None or self.underFlow is None:
            return None
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = None
        if not os.path.exists(folder):
            os.makedirs(folder)
        pathNImage = folder+'/'+str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath + '_' + filetype + '.bmp'
        if filetype == 'overflow':
            self.overFlow = scaleValues(0,255,self.overFlow)
            imageSaveDLP = im.fromarray(self.overFlow)
        else:
            self.underFlow = scaleValues(0, 255, self.underFlow)
            imageSaveDLP = im.fromarray(self.underFlow)

        imageSaveDLP.convert('RGB').save(pathNImage)

    def isFileExists(self,pathImage,filetype='png'):
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        pathNImage = pathImage + '/' + str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath + '.' +filetype
        return os.path.exists(pathNImage)