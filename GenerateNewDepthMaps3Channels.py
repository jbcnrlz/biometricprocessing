from baseClasses.PreProcessingStep import *
import math, numpy as np,matlab.engine, os
from scipy.signal import savgol_filter

class GenerateNewDepthMaps3Channels(PreProcessingStep):

    def saveTXTMatlab(self,path,face):
        f = open(path,'w')
        for p in face:
            f.write(' '.join(map(str,p)) + '\n')
        f.close()

    def generateImage(self,path):        
        eng = matlab.engine.start_matlab()
        return np.array(eng.generateNewDepthMapsXYZ(path)).tolist()

    def doPreProcessing(self,template):
        subject = "%04d" % (template.itemClass)
        template3Dobj = template.rawRepr.split(os.path.sep)[:-3]
        folderType = template3Dobj[template3Dobj.index(subject) + 1]
        txtFilePath = os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'_processing_matlab.obj')
        self.saveTXTMatlab(txtFilePath,template.image)
        template.image = savgol_filter(np.array(self.generateImage(txtFilePath)),51,3).tolist()
        os.remove(txtFilePath)
        return template
