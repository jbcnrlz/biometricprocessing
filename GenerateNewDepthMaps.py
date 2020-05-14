from baseClasses.PreProcessingStep import *
import math, numpy as np, os, matlab.engine
from scipy.signal import savgol_filter
from helper.functions import scaleValues
from IIITDTemplate import *

class GenerateNewDepthMaps(PreProcessingStep):

    def saveTXTMatlab(self,path,face):
        f = open(path,'w')
        for p in face:
            f.write(' '.join(map(str,p)) + '\n')
        f.close()

    def generateImage(self,path):        
        eng = matlab.engine.start_matlab()
        return np.array(eng.generateNewDepthMaps(path,100,100)).tolist()


    def doPreProcessing(self,template):
        if '3DObj' in template.rawRepr:
            subject = "%04d" % (template.itemClass)
            template3Dobj = template.rawRepr.split(os.path.sep)[:-2]
            folderType = template3Dobj[template3Dobj.index(subject) + 1]
        elif not type(template) is IIITDTemplate:
            subject = "%04d" % (template.itemClass)
            template3Dobj = template.rawRepr.split(os.path.sep)[:-3]
            folderType = template3Dobj[template3Dobj.index(subject) + 1]
        elif type(template) is IIITDTemplate:
            txtFilePath = template.rawRepr[:-4]+ '_processing_matlab.obj'
            self.saveTXTMatlab(txtFilePath,template.image)
            #template.image = savgol_filter(np.array(self.generateImage(txtFilePath)),51,3).tolist()
            template.image = np.array(self.generateImage(txtFilePath))
            template.image = scaleValues(0, 255, template.image)
            template.save(True)
            os.remove(txtFilePath)
            return template

        txtFilePath = os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'_processing_matlab.obj')
        self.saveTXTMatlab(txtFilePath,template.image)
        #template.image = savgol_filter(np.array(self.generateImage(txtFilePath)),51,3).tolist()
        template.image = np.array(self.generateImage(txtFilePath))
        template.image = scaleValues(0, 255, template.image)
        template.save()
        os.remove(txtFilePath)
        return template
