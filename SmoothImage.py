from baseClasses.PreProcessingStep import *
import math, numpy as np,matlab.engine, os
from scipy.signal import savgol_filter

class SmoothImage(PreProcessingStep):

    def doPreProcessing(self,template):
        template.image = savgol_filter(np.array(template.image),51,3).tolist()
        return template
