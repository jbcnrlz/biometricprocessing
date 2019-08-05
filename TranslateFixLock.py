from baseClasses.PreProcessingStep import *
import numpy as np
from helper.functions import getRotationMatrix3D

class TranslateFixLock(PreProcessingStep):

    def __init__(self,**kwargs):
        self.nosetipindex = kwargs.get('nosetip',2)
        if type(self.nosetipindex) is not int and self.nosetipindex.isdigit():
            self.nosetipindex = int(self.nosetipindex)

    def doPreProcessing(self,template):
        template.image = template.image - template.faceMarks[self.nosetipindex]

        rm90cw = getRotationMatrix3D(np.deg2rad(90),'z')
        ones = np.ones((template.image.shape[0], 1))
        homo = np.concatenate((template.image, ones),axis=1)
        template.image = np.dot(homo, rm90cw)[:,:-1]

        rm90cw = getRotationMatrix3D(np.deg2rad(180),'y')
        ones = np.ones((template.image.shape[0], 1))
        homo = np.concatenate((template.image, ones),axis=1)
        template.image = np.dot(homo, rm90cw)[:,:-1]

        return template
