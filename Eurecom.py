from baseClasses.DatabaseProcessingUtility import *
from helper.functions import outputObj,loadOBJ,minmax
from TemplateNode import *
from SymmetricFilling import *
from FixWithAveragedModel import *
from RotateFace import *
import os, glob, copy, numpy as np

class EurecomKinect(DatabaseProcessingUtility):

    def saveTemplateImage(self):
        for t in self.templates:
            t.save(True)
            t.outputMarks(True)

    def loadTemplateImage(self):
        for t in self.templates:
            t.loadImage()        

    def loadSymmFilledImages(self):
        for t in self.templates:
            t.loadSymFilledImage()        

    def getImageType(self):
        return self.__imageType

    def setImageType(self,value):
        if (value == 'Depth'):
            self.__imageType = os.path.join('Depth','DepthBMP')
        else:
            self.__imageType = value

    imageType = property(getImageType,setImageType)    

    def __init__(self,path,currentType,imageType,filePose=['Neutral']):
        super().__init__(path)
        self.folderTemplate = currentType
        self.imageType = imageType
        self.filePose = filePose

    def getFilesFromDirectory(self,path,filePoseCurrent):
        if (self.imageType == '3DObj'):
            return glob.glob(path + '/*'+filePoseCurrent+'.obj')
        else:
            return glob.glob(path + '/*'+filePoseCurrent+'.bmp')

    def getDirecotiresInPath(self,path):
        return [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]

    def getTemplateType(self,fileName):
        explodedFile = fileName.split('.')
        explodedFile = explodedFile[0].split('_')
        return explodedFile.pop()

    def feedTemplates(self,lazy=False):
        for subject in self.getDirecotiresInPath(self.databasePath):
            if (os.path.exists(os.path.join(self.databasePath,subject,self.folderTemplate))):
                directories = []
                for f in self.filePose:
                    directories += self.getFilesFromDirectory(os.path.join(self.databasePath,subject,self.folderTemplate,self.imageType),f)
                for d in directories:
                    euTemp = EurecomTemplate(os.path.join(self.databasePath,subject,self.folderTemplate,self.imageType,d),self.getTemplateType(d),lazy)
                    euTemp.itemClass = subject
                    euTemp.folderTemplate = self.folderTemplate
                    euTemp.loadMarks(self.imageType)
                    self.templates.append(euTemp)

    def getDatabaseRepresentation(self):
        returnRepr = [[],[]]
        for template in self.templates:
            returnRepr[0].append(template.features)
            returnRepr[1].append(template.itemClass)
        return returnRepr

    def getTemplateFromSubject(self,subject):
        returnTemplates = []
        for template in self.templates:
            if (template.itemClass == subject):
                returnTemplates.append(template)
        return returnTemplates

    def generateAverageFaceModel(self):
        sm = SymmetricFilling()
        averagedFace = self.templates[0].image
        for template in self.templates[1:3]:
            templateImage = sm.applyICP(averagedFace,np.array(template.image))
            averagedFace = sm.symmetricFillingPCL(np.array(averagedFace,dtype=np.float32),templateImage).tolist()

        outputObj(averagedFace,os.path.join('averagedFace.obj'))

    def fixingProbe(self):
        fwa = FixWithAveragedModel()
        for template in self.templates:
            template = fwa.doPreProcessing(template)


    def loadNewDepthImage(self):
        for t in self.templates:
            t.loadNewDepthImage()

    def loadRotatedFaces(self,angles):
        addTemplates = []
        for t in self.templates:
            print("Gerando copia de "+t.rawRepr)
            for a in angles:
                print("Angulo = "+ str(a))
                nobj = copy.deepcopy(t)
                with im.open(nobj.rawRepr[0:-4] + '_rotate_'+str(a)+'_newdepth.bmp') as currImg:
                    nobj.image = np.asarray(currImg)
                    nobj.rawRepr = nobj.rawRepr[0:-4] + '_rotate_'+str(a)+'_newdepth.bmp'
                addTemplates.append(nobj)

        self.templates = self.templates + addTemplates

    def loadRotatedFacesPC(self,angles):
        addTemplates = []
        rface = RotateFace()
        for t in self.templates:
            print("Gerando copia de "+t.rawRepr)
            for a in angles:
                print("Angulo = "+ str(a))
                rty = rface.getRotationMatrix(a,'y')
                nobj = copy.deepcopy(t)
                pathBroken = nobj.rawRepr.split(os.path.sep)
                fileName = pathBroken[-1]
                pathBroken = os.path.sep.join(pathBroken[:-2])
                fileName = fileName.split('_')
                if (fileName[-1] == 'symmetricfilled.obj'):
                    fileName = '_'.join(fileName[0:-1])
                else:
                    fileName = '_'.join(fileName)[0:-4]
                pathBroken = os.path.join(pathBroken,'Depth','DepthBMP',fileName+'_rotate_'+str(a)+'.obj')
                
                nobj.rawRepr = pathBroken
                nobj.faceMarks = rface.multiplyMatrices(nobj.faceMarks,rty)
                addTemplates.append(nobj)

        self.templates = self.templates + addTemplates

    def generateDatabaseFile(self,path,faces,otherBases=None,outputDesired='SVMTorchFormat',excludeFiles=[]):
        currSearch = self.templates
        if otherBases:
            for o in otherBases:
                currSearch = currSearch + o.templates

        currFiles = []
        for f in currSearch:
            originalPathFile = f.rawRepr.split(os.path.sep)
            originalPathFile = originalPathFile[-1].split('_')[-1]
            print(originalPathFile)
            print(excludeFiles)
            if not (originalPathFile in excludeFiles):
                print('Entrou')
                if (f.typeTemplate in faces):
                    for ss in range(len(faces[f.typeTemplate])):
                        if f.folderTemplate == faces[f.typeTemplate][ss]:
                            currFiles.append(f)

        return getattr(self,outputDesired)(currFiles,path)

    def SVMTorchFormat(self,data,path):
        f = open(path,'w')
        f.write(str(len(data))+' '+str(len(data[0].features) + 1) + '\n')
        for t in data:
            print("Imprimindo linha")
            f.write(' '.join([str(e) for e in t.features]) + ' ' + str(t.itemClass - 1) + '\n')
        f.close()
        return []

    def caffeFormat(self,data,path):
        f = open(path,'w')
        for t in data:
            f.write(' '.join([str(e) for e in t.features]) + '\n')
        f.close()
        return []

    def h5Format(self,data,path):
        import h5py
        X = np.zeros((len(data),len(data[0].features)))
        Y = np.zeros((len(data),1))
        for t in range(len(data)):
            X[t] = data[t].features
            Y[t] = data[t].itemClass

        with h5py.File(path,'w') as H:
            H.create_dataset( 'X', data=X )
            H.create_dataset( 'Y', data=Y )
        with open(path[:-3]+'_list.txt','w') as L:
            L.write( path )

    def imageForCaffe(self,data,path):
        dataNum = 0
        for t in data:
            dataNum = dataNum + 1
            fixedFeature = np.array(t.features).reshape(int(len(t.features)/14),14)
            f = open(os.path.join(path,str(t.itemClass)+'_'+t.typeTemplate+'_'+str(dataNum)+'.txt'),'w')            
            for i in fixedFeature:
                finalPrint = ''
                for j in i:
                    finalPrint = finalPrint + str(j) + " "
                f.write(finalPrint + '\n')
            f.close()

        return []

    def generateCharsClasses(self,data,path):
        returnArray = [[],[]]
        for t in data:
            returnArray[0].append(t.features)
            returnArray[1].append(t.itemClass)

        return returnArray

    def noiseImageGenerate(self,noiseQntde=0.4,imageNoise = 4):
        addTemplates = []
        for t in self.templates:
            print("Gerando copia de "+t.rawRepr)
            for t in self.templates:
                if 'rotate' not in t.rawRepr:
                    print("Gerando imagem ruidosa de " + t.rawRepr)
                    for iq in range(imageNoise):
                        print("Fazendo imagem " + str(iq))
                        nobj = copy.deepcopy(t)
                        nobj.image = self.get_white_noise_image(nobj.image)
                        addTemplates.append(nobj)


        self.templates = self.templates + addTemplates

    def get_white_noise_image(self,imageNoise,noiseQntde = 0.1):
        arrayImage = np.asarray(imageNoise).copy()

        noiseQntde = (arrayImage.shape[0] * arrayImage.shape[1]) * noiseQntde
        noiseQntde = int(noiseQntde)
        print("Quantidade de ruido = "+str(noiseQntde))    

        while (noiseQntde > 0):
            x = int(random.random() * arrayImage.shape[0])
            y = int(random.random() * arrayImage.shape[1])
            arrayImage[x,y] = int(random.random() * 255)
            noiseQntde = noiseQntde - 1

        return im.fromarray(arrayImage)

    def normalizeData(self):
        for t in self.templates:
            print(t.features)
            t.features = minmax(t.features)
            print(t.features)

if __name__ == '__main__':
    e = EurecomKinect('/home/jbcnrlz/Documents/eurecom/EURECOM_Kinect_Face_Dataset','s1','Depth')
    e.feedTemplates()