from math import sqrt
from baseClasses.Template import *
from helper.functions import outputObj, loadOBJ
from PIL import Image as im
import os, numpy as np, random, pcl

class LFWTemplate(Template):

    folderTemplate = None
    faceMarks = []
    layersChar = None

    def __init__(self,pathFile,subject,lazyData=False,dataset=None):
        self.itemClass = subject
        super().__init__(pathFile,None,lazyData,dataset)

    def save(self,saveOnPath=False):
        if (not saveOnPath):
            if (not os.path.exists('temporaryTemplate')):
                os.makedirs('temporaryTemplate')

            outputObj(self.image,os.path.join('temporaryTemplate',str(self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.obj'))
            self.outputMarks()
        else:
            self.image.save(self.rawRepr[0:-4] + '_newdepth.bmp')


    def loadMarks(self,typeTemplate='Depth'):
        self.faceMarks = []


    def loadNewDepthImage(self):
        self.image = im.open(self.rawRepr[0:-4] + '_newdepth.bmp')
        self.loadMarks('newdepth')

    def saveImageTraining(self,avgImageSave=True,pathImage='generated_images_lbp'):
        #imageSaveDLP = np.array(self.layersChar)
        if (avgImageSave):
            avImage = np.zeros((self.layersChar.shape[0],self.layersChar.shape[1]))        
            for i in range(self.layersChar.shape[0]):
                for j in range(self.layersChar.shape[1]):
                    avImage[i,j] = self.layersChar[i,j,0] + self.layersChar[i,j,1] + self.layersChar[i,j,2] + self.layersChar[i,j,3]
                    avImage[i,j] = avImage[i,j] / 4
            avImage = im.fromarray(np.uint8(avImage))
            avImage.save(pathImage+'/averageImage/'+str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath +'.jpg')
        
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = im.fromarray(np.uint8(self.layersChar))
        pathNImage = pathImage+'/'+str(self.itemClass) + '_' + fullPath +'.png'
        while (os.path.exists(pathNImage)):
            idxRandomIm = random.randint(1,255)
            pathNImage = pathImage+'/'+str(self.itemClass) + '_' + fullPath +'_'+str(idxRandomIm)+'.png'
            
        print("Gerando imagem de "+pathNImage)
        imageSaveDLP.save(pathNImage)
        return pathNImage