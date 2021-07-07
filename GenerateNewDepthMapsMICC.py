from baseClasses.PreProcessingStep import *
from helper.functions import scaleValues
import math, numpy as np,matlab.engine, os, cv2
from scipy.signal import savgol_filter
from helper.functions import outputObj
from scipy.ndimage import gaussian_filter

class GenerateNewDepthMapsMICC(PreProcessingStep):

    def saveTXTMatlab(self,path,face):
        
        f = open(path,'w')
        for p in face:
            f.write(','.join(map(str,p)) + '\n')
        f.close()        

    def generateImage(self,path,x=100,y=100):        
        eng = matlab.engine.start_matlab()
        return np.array(eng.generateNewDepthMaps(path,x,y)).tolist()

    def fixImage(self,t):
        maxValue = abs(t.image.max() - 255)
        for i in range(t.image.shape[0]):
            for j in range(t.image.shape[1]):
                t.image[i,j] = int(t.image[i,j] + maxValue)

        return t

    def doPreProcessing(self,template):
        txtFilePath = template.rawRepr[0:-4] + '_processing_matlab.obj'
        self.saveTXTMatlab(txtFilePath,template.image)
        template.image = np.array(self.generateImage(txtFilePath,100,100))
        if template.resolution == 'low':
            template.image = gaussian_filter(template.image,2)
        template.image = scaleValues(0, 255, template.image)
        template.saveNewDepth()
        os.remove(txtFilePath)
        return template
