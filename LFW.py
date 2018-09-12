from baseClasses.DatabaseProcessingUtility import *
from helper.functions import outputObj,loadOBJ,minmax
from LFWTemplate import *
from SymmetricFilling import *
from FixWithAveragedModel import *
from RotateFace import *
from models.models import *
from models.engine_creation import *
import os, glob, copy, numpy as np

class LFW(DatabaseProcessingUtility):

    databaseName = 'LFW'

    def getModel(self):
        dts = ss.query(Database).filter(Database.nome==self.databaseName).first()
        if (not dts):
            dts = Database(nome=self.databaseName,url_disp='http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html')
            ss.add(dts)
            ss.commit()
            
        return dts

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

    def __init__(self,path):
        super().__init__(path)

    def getFilesFromDirectory(self,path,typeImage):        
        return glob.glob(path + '/*.'+typeImage)

    def getDirecotiresInPath(self,path):
        return [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]

    def getTemplateType(self,fileName):
        explodedFile = fileName.split('.')
        explodedFile = explodedFile[0].split('_')
        return explodedFile.pop()

    def feedTemplates(self,lazy=False,typeFile='obj',innerFolders=['probe','gallery'],fileName='face'):
        itemClass = 1
        for subject in self.getDirecotiresInPath(self.databasePath):
            for infold in innerFolders:
                euTemp = ''
                print("Carregando " + subject)
                if (typeFile == 'obj'):
                    euTemp = LFWTemplate(os.path.join(self.databasePath,subject,infold,fileName+'.obj'),itemClass,lazy,self)
                elif(typeFile == 'bmp'):                    
                    euTemp = LFWTemplate(os.path.join(self.databasePath,subject,infold,fileName+'_newdepth.bmp'),itemClass,lazy,self)
                elif(typeFile == 'jpg'):
                    imagens = self.getFilesFromDirectory(os.path.join(self.databasePath,subject,infold),'jpg')
                    for i in imagens:
                        euTemp = LFWTemplate(i,itemClass,lazy,self)
                elif (typeFile == 'pcd'):
                    euTemp = LFWTemplate(os.path.join(self.databasePath,subject,infold,fileName+'.pcd'),itemClass,lazy,self)
                elif (typeFile == 'spcd'):
                    euTemp = LFWTemplate(os.path.join(self.databasePath,subject,infold,fileName+'_segmented.pcd'),itemClass,lazy,self)
                self.templates.append(euTemp)
            itemClass += 1

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

    def loadRotatedFaces(self,angles,axis):
        addTemplates = []
        for t in self.templates:
            print("Gerando copia de "+t.rawRepr)
            for ax in axis:
                for a in angles:
                    print("Angulo = "+ str(a))
                    nobj = copy.deepcopy(t)
                    pathImages = nobj.rawRepr.split(os.path.sep)
                    pathImages = pathImages[:-1]
                    pathImages = os.path.sep.join(pathImages)
                    if os.path.exists(os.path.join(pathImages,'face_segmented_rotate_'+str(a)+'_newdepth.bmp')):
                        pathImages = os.path.join(pathImages,'face_segmented_rotate_'+str(a)+'_newdepth.bmp')
                    else:
                        pathImages = os.path.join(pathImages,'face_rotate_'+str(a)+'_'+ax+'_newdepth.bmp')
                    with im.open(pathImages) as currImg:
                        nobj.image = np.asarray(currImg)
                        nobj.rawRepr = pathImages
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

    def generateDatabaseFile(self,path,otherBases=None,outputDesired='SVMTorchFormat'):
        currSearch = self.templates
        if otherBases:
            for o in otherBases:
                currSearch = currSearch + o.templates
        currFiles = []
        filesPath = []
        for f in currSearch:
            currFiles.append(f)
            filesPath.append(f.rawRepr)

        pathFaces = path[:-3]
        pathFaces = pathFaces + '_faces.txt'
        f = open(pathFaces,'w')
        for t in filesPath:
            print("Imprimindo linha arquivos")
            f.write(t + '\n')
        f.close()
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