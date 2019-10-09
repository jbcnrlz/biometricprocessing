from baseClasses.PreProcessingStep import *
from SymmetricFilling import *
import math, numpy as np, os
from scipy.signal import savgol_filter
from helper.functions import loadOBJ

class FixWithAveragedModel(PreProcessingStep):

    def doPreProcessing(self,template):
        sm = SymmetricFilling()
        a,b,c,d = loadOBJ(os.path.join('temporaryTemplate',str(template.itemClass) + '_s1_' + template.typeTemplate + '.obj'))
        template.image = sm.applyICP(c,np.array(template.image))
        return template
