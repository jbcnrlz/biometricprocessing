import numpy as np, math, operator, os, logging, scipy.ndimage as ndimage, matlab.engine
from baseClasses.BiometricProcessing import *
from helper.functions import generateHistogram, loadOBJ, generateHistogramUniform, bilinear_interpolation, mergeArraysDiff
from sklearn import svm
from helper.functions import minmax
from PIL import Image as im
from scipy.spatial.distance import *
from scipy.special import expit
from mahotas.interpolate import shift

class GenerateProfiles(BiometricProcessing):

    def __init__(self,database):
        self.databases = database

    def setupTemplate(self,template):
        
        template.loadMarks('3DObj')
        subject = "%04d" % (template.itemClass)
        template3Dobj = template.rawRepr.split(os.path.sep)[:-3]
        folderType = template3Dobj[template3Dobj.index(subject) + 1]
        a, b, c, y = loadOBJ(os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'.obj'))
        template.image = c

        return template

    def cleanupTemplate(self,template):
        template.image = im.fromarray(np.array(template.image,dtype=np.uint8))
        template.image = template.image.rotate(-180)        
        return template