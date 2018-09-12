import numpy as np, math, operator, os, logging, scipy.ndimage as ndimage, matlab.engine, pywt, cv2
from baseClasses.BiometricProcessing import *
from helper.functions import generateHistogram, loadOBJ, generateHistogramUniform, bilinear_interpolation, mergeArraysDiff, minmax
from PIL import Image as im
from scipy.spatial.distance import *
from scipy.special import expit
from mahotas.interpolate import shift
from mahotas.lbp import lbp
from LFWTemplate import *
from skimage import feature

class MahotasLBP(BiometricProcessing):

    def __init__(self,windowsize,binsize,database):
        self.windowSize = windowsize
        self.binsize = binsize
        self.databases = database

    def generateCode(self,image,center,typeOp='Normal'):
        idxs = [
            (center[0]-1,center[1]-1),
            (center[0]-1,center[1]),
            (center[0]-1,center[1]+1),
            (center[0],center[1]-1),
            (center[0],center[1]+1),
            (center[0]+1,center[1]-1),
            (center[0]+1,center[1]),
            (center[0]+1,center[1]+1)
        ]
        layers = [[],[],[],[]]
        for i in idxs:
            #subraction = int(round(image[i[0]][i[1]] - image[center[0]][center[1]]))
            #if subraction < -7:
            #    subraction = -7
            #elif subraction > 7:
            #    subraction = 7

            subraction = 0
            if typeOp == 'Normal':
                subraction = int(round(image[i[0]][i[1]] - image[center[0]][center[1]]))
                if subraction < -7:
                    subraction = -7
                elif subraction > 7:
                    subraction = 7
            else:
                subraction = image[i[0]][i[1]] - image[center[0]][center[1]]
                subraction = np.histogram(expit(subraction),bins=7,range=[0,1] )[0]                
                subraction = np.argwhere(subraction==1)[0][0]
            bin = '{0:03b}'.format(abs(subraction))
            layers[0].append(str(int(subraction >= 0)))
            layers[1].append(bin[0])
            layers[2].append(bin[1])
            layers[3].append(bin[2])
        for l in range(len(layers)):
            layers[l] = int(''.join(layers[l]),2)       
        return layers

    def cropImage(self,image,le,re,no):
        distance = list(map(operator.sub,le[:2],re[:2]))
        distance = int(math.sqrt((distance[0]**2) + (distance[1]**2)))
        points_crop = (no[0] - distance,no[1] - distance,no[0] + distance,no[1] + distance)
        points_crop = list(map(int,points_crop))   
        return image.crop((points_crop[0],points_crop[1],points_crop[2],points_crop[3]))

    def setupTemplate(self,template):
        if (not type(template) is LFWTemplate):
            template.loadMarks('3DObj')
            subject = "%04d" % (template.itemClass)
            template3Dobj = template.rawRepr.split(os.path.sep)[:-3]
            folderType = template3Dobj[template3Dobj.index(subject) + 1]
            a, b, c, y = loadOBJ(os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'.obj'))
            template.image = c

        return template

    def cleanupTemplate(self,template):
        template.layersChar = np.zeros((len(template.image),len(template.image[0]),4))
        template.image = im.fromarray(np.array(template.image,dtype=np.uint8))
        template.image = template.image.rotate(-180)
        template.save(True)
        return template

    def getFace(self,template):
        img = np.array(template.image)
        foundFace = []
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            foundFace.append(gray[y:y+h, x:x+w])

        return foundFace

    def featureExtraction(self):
        print("Iniciando feature extraction")        
        for database in range(len(self.databases)):
            newDb = []
            for template in self.databases[database].templates:
                imgFace = self.getFace(template)

                if (len(imgFace) > 0):
                    imgCroped = imgFace[0]
                    offsetx = int(math.ceil(imgCroped.shape[0] / float(self.windowSize)))
                    offsety = int(math.ceil(imgCroped.shape[1] / float(self.windowSize)))
                    fullImageDescriptor = []
                    print('Gerando descritor de: '+str(template.itemClass))
                    for i in range(0,imgCroped.shape[0],offsetx):
                        for j in range(0,imgCroped.shape[1],offsety):
                            #fullImageDescriptor += lbp(imgCroped[i:(i+offsetx),j:(j+offsety)],1,8).tolist()
                            lbp = feature.local_binary_pattern(imgCroped[i:(i+offsetx),j:(j+offsety)],8,1,method="uniform")
                            (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 11),range=(0, 10))
                            hist = minmax(hist)
                            fullImageDescriptor += hist

                    template.features = fullImageDescriptor
                    newDb.append(template)

            self.databases[database].templates = newDb
