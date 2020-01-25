from baseClasses.PreProcessingStep import *
from helper.functions import scaleValues
import math, numpy as np,matlab.engine, os
from scipy.signal import savgol_filter
from helper.functions import outputObj

class GenerateNewDepthMapsLFW(PreProcessingStep):

    def saveTXTMatlab(self,path,face):
        
        face = np.array(face)
        idxFace = face[:,2].tolist().index(max(face[:,2]))
        face = face - face[idxFace]
        face = face / 1000.0
        #print(face[idxFace])
        #input()
        f = open(path,'w')
        #f.write('g \n')#
        for p in face:
            f.write(' '.join(map(str,p)) + '\n')
            #f.write('v ' + ' '.join(map(str,p)) + '\n')#
        #f.write('g 1 \n')#
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
        #template.image = savgol_filter(np.array(self.generateImage(txtFilePath,100,100)),51,3).tolist()
        template.image = np.array(self.generateImage(txtFilePath,100,100))
        template.image = scaleValues(0, 255, template.image)
        template.saveNewDepth()
        os.remove(txtFilePath)
        return template
