from baseClasses.DatabaseProcessingUtility import *
from helper.functions import outputObj,loadOBJ,minmax
from IIITDTemplate import *
from FixWithAveragedModel import *
from RotateFace import *
import os, glob, copy, numpy as np

class IIITDKinectDataset(DatabaseProcessingUtility):

    def loadTemplateImage(self):
        for t in self.templates:
            t.loadImage()

    def loadRotatedFaces(self,angles,axis):
        addTemplates = []
        for t in self.templates:
            print("Gerando copia de "+t.rawRepr)
            for ax in axis:
                for a in angles:
                    print("Angulo = "+ str(a))
                    nobj = copy.deepcopy(t)
                    if os.path.exists(nobj.rawRepr[0:-4] +'_rotate_'+str(a)+'_'+ax+'_newdepth.bmp'):
                        pathImages = nobj.rawRepr[0:-4] +'_rotate_'+str(a)+'_'+ax+'_newdepth.bmp'
                        with im.open(pathImages) as currImg:
                            nobj.image = np.asarray(currImg)
                            nobj.rawRepr = pathImages
                        addTemplates.append(nobj)

        self.templates = self.templates + addTemplates

    def __init__(self,path,type='Range'):
        self.imageType = type
        super().__init__(path)

    def loadRotatedFaces(self,angles,axis):
        addTemplates = []
        for t in self.templates:
            print("Gerando copia de "+t.rawRepr)
            for ax in axis:
                if ax == 'xy':
                    for a in angles:
                        for b in angles:
                            if a == 0 or b == 0:
                                continue
                            print("Cross Angulo "+ str(a) + ' '+str(b))
                            nobj = copy.deepcopy(t)
                            if os.path.exists(nobj.rawRepr[0:-4] +'_rotate_'+str(a)+'_'+str(b)+'_'+ax+'_newdepth.bmp'):
                                pathImages = nobj.rawRepr[0:-4] +'_rotate_'+str(a)+'_'+str(b)+'_'+ax+'_newdepth.bmp'
                                with im.open(pathImages) as currImg:
                                    nobj.image = np.asarray(currImg)
                                    nobj.rawRepr = pathImages
                                addTemplates.append(nobj)
                else:
                    for a in angles:
                        print("Angulo = "+ str(a))
                        nobj = copy.deepcopy(t)
                        if os.path.exists(nobj.rawRepr[0:-4] +'_rotate_'+str(a)+'_'+ax+'_newdepth.bmp'):
                            pathImages = nobj.rawRepr[0:-4] +'_rotate_'+str(a)+'_'+ax+'_newdepth.bmp'
                            try:
                                with im.open(pathImages) as currImg:
                                    nobj.image = np.asarray(currImg)
                                    nobj.rawRepr = pathImages
                                addTemplates.append(nobj)
                            except:
                                continue

        self.templates = self.templates + addTemplates


    def loadNewDepthImage(self):
        for t in self.templates:
            t.loadNewDepthImage()

    def getFilesFromDirectory(self,path):
        return glob.glob(path + '/*.'+self.extensionFile[self.imageType])

    def getDirecotiresInPath(self,path):
        return [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]

    def getTemplateType(self,fileName):
        explodedFile = fileName.split('.')
        explodedFile = explodedFile[0].split('_')
        return explodedFile.pop()

    def feedTemplates(self):
        for subject in self.getDirecotiresInPath(self.databasePath):
            if self.imageType in ['Depth','Range']:
                directories = self.getFilesFromDirectory(os.path.join(self.databasePath,subject,'Depth','depthnocolor'))
            else:
                directories = self.getFilesFromDirectory(os.path.join(self.databasePath,subject,'Depth'))
            for d in directories:
                statinfo = os.stat(os.path.join(self.databasePath,subject,d))
                if statinfo.st_size > 15 and 'rotate' not in d and 'matlab' not in d and 'symfilled' not in d:
                    euTemp = IIITDTemplate(os.path.join(self.databasePath,subject,d),subject)
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

    def fixingProbe(self):
        fwa = FixWithAveragedModel()
        for template in self.templates:
            template = fwa.doPreProcessing(template)

    def generateDatabaseFile(self,path,otherBases=None,outputDesired='SVMTorchFormat',excludeFiles=[]):
        currSearch = self.templates
        if otherBases:
            for o in otherBases:
                currSearch = currSearch + o.templates
        currFiles = []
        filesPath = []
        for f in currSearch:
            currFiles.append(f)
            filesPath.append(f.rawRepr)

        pathFaces = path[:-4]
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
            if not (t.features is None):
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

    def normalizeData(self):
        for t in self.templates:
            print(t.features)
            t.features = minmax(t.features)
            print(t.features)