from baseClasses.PreProcessingStep import *
import math, numpy as np,matlab.engine, os
from PIL import Image as im
from scipy.signal import savgol_filter
from helper.functions import outputObj

class GenerateNewDepthMapsRFRGC(PreProcessingStep):

    checkForCloud = True
    fileExtension = ''

    def __init__(self, **kwargs):
        self.regenarate = kwargs.get('regenarate', True)

    def saveTXTMatlab(self,path,face):
        f = open(path,'w')
        for p in face:
            f.write(' '.join(map(str,p)) + '\n')
        f.close()

    def generateImage(self,path,x=100,y=100):        
        eng = matlab.engine.start_matlab()
        return np.array(eng.generateNewDepthMaps(path,x,y)).tolist()

    def generateCloud(self,template):
        imageFace = []
        for i in range(template.image.shape[0]):
            for j in range(template.image.shape[1]):
                if (template.image[i][j] > 40):
                    imageFace.append([i,j,template.image[i][j]])

        return np.array(imageFace)

    def doPreProcessing(self,template):
        if self.regenarate or not os.path.exists(template.rawRepr[0:-4]+self.fileExtension+'_newdepth.jpeg'):
            template3Dobj = template.rawRepr.split(os.path.sep)[:-1]
            fileName = template.rawRepr.split(os.path.sep)[-1][:-4]
            txtFilePath = os.path.join(os.path.sep.join(template3Dobj),fileName+'_processing_matlab.txt')
            newImage = None
            if (not self.checkForCloud) or (template.rawRepr[-3:] == 'obj'):
                newImage = template.image
            else:
                newImage = self.generateCloud(template)
            self.saveTXTMatlab(txtFilePath,newImage)
            template.image = np.array(self.generateImage(txtFilePath)).T
            #template.image = savgol_filter(template.image,51,3)
            template.saveNewDepth(self.fileExtension)
            os.remove(txtFilePath)
        else:
            template.image = np.array(im.open(template.rawRepr[0:-4]+self.fileExtension+'_newdepth.jpeg'))
        return template
