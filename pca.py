import numpy as np, math, operator, os, logging, scipy.ndimage as ndimage, matlab.engine, pywt, cv2
from baseClasses.BiometricProcessing import *
from helper.functions import generateHistogram, loadOBJ, generateHistogramUniform, bilinear_interpolation, mergeArraysDiff, minmax
from PIL import Image as im
from LFWTemplate import *
from skimage import feature
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAImpl(BiometricProcessing):

    def __init__(self,windowsize,binsize,database):
        self.windowSize = windowsize
        self.binsize = binsize
        self.databases = database

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
        #pca = PCA(n_components='mle', svd_solver='full')
        print("Iniciando feature extraction")        
        for database in range(len(self.databases)):
            newDb = []
            for template in self.databases[database].templates:
                imgFace = self.getFace(template)

                if (len(imgFace) > 0):
                    imgCroped = cv2.resize(imgFace[0],(60,60))
                    imgCroped = StandardScaler().fit_transform(imgCroped)
                    fullImageDescriptor = []
                    print('Gerando descritor de: '+str(template.itemClass))
                    template.features = imgCroped.flatten()
                    newDb.append(template)

            self.databases[database].templates = newDb
