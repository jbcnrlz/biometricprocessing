from baseClasses.DatabaseProcessingUtility import *
from BosphorusTemplate import *
from helper.functions import outputObj, loadOBJ, minmax, getDirectoriesInPath
from TemplateNode import *
from RotateFace import *
import os, glob, copy, numpy as np


class Bosphorus(DatabaseProcessingUtility):

    def saveNewDepthImages(self):
        for t in self.templates:
            t.saveNewDepth()

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

    def setImageType(self, value):
        if (value == 'Depth'):
            self.__imageType = os.path.join('Depth', 'DepthBMP')
        else:
            self.__imageType = value

    imageType = property(getImageType, setImageType)

    def __init__(self, path, imageType, filePose=['N']):
        super().__init__(path)
        self.imageType = imageType
        self.filePose = filePose

    def getFilesFromDirectory(self,path):
        returnFiles = []
        for f in glob.glob(path + '/*.'+self.extensionFile[self.imageType]):
            returnFiles.append(f)

        return returnFiles

    def getTemplateType(self, fileName):
        explodedFile = fileName.split('.')
        explodedFile = explodedFile[0].split('_')
        return explodedFile.pop()

    def feedTemplatesFromList(self,listOfFaces):
        for l in listOfFaces:
            subjectNumber = l.split(os.path.sep)[-1]
            subjectNumber = int(subjectNumber[2:5])
            euTemp = BosphorusTemplate(l, subjectNumber)
            self.templates.append(euTemp)

    def feedTemplates(self):
        for subject in getDirectoriesInPath(self.databasePath):
            directories = self.getFilesFromDirectory(os.path.join(self.databasePath,subject))
            for d in directories:
                for ft in self.filePose:
                    if ft in d:
                        subjectNumber = int(subject[-3:])
                        euTemp = BosphorusTemplate(os.path.join(self.databasePath,subject,d),subjectNumber)
                        self.templates.append(euTemp)
                        break

    def getDatabaseRepresentation(self):
        returnRepr = [[], []]
        for template in self.templates:
            returnRepr[0].append(template.features)
            returnRepr[1].append(template.itemClass)
        return returnRepr

    def getTemplateFromSubject(self, subject):
        returnTemplates = []
        for template in self.templates:
            if (template.itemClass == subject):
                returnTemplates.append(template)
        return returnTemplates

    def loadNewDepthImage(self):
        for t in self.templates:
            t.loadNewDepthImage()

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
                            if os.path.exists(nobj.rawRepr[0:-13] +'_rotate_'+str(a)+'_'+str(b)+'_'+ax+'_newdepth.bmp'):
                                pathImages = nobj.rawRepr[0:-13] +'_rotate_'+str(a)+'_'+str(b)+'_'+ax+'_newdepth.bmp'
                                with im.open(pathImages) as currImg:
                                    nobj.image = np.asarray(currImg)
                                    nobj.rawRepr = pathImages
                                addTemplates.append(nobj)
                else:
                    for a in angles:
                        print("Angulo = "+ str(a))
                        nobj = copy.deepcopy(t)
                        if os.path.exists(nobj.rawRepr[0:-13] +'_rotate_'+str(a)+'_'+ax+'_newdepth.bmp'):
                            pathImages = nobj.rawRepr[0:-13] +'_rotate_'+str(a)+'_'+ax+'_newdepth.bmp'
                            with im.open(pathImages) as currImg:
                                nobj.image = np.asarray(currImg)
                                nobj.rawRepr = pathImages
                            addTemplates.append(nobj)

        self.templates = self.templates + addTemplates

    def loadRotatedFacesPC(self, angles):
        addTemplates = []
        rface = RotateFace()
        for t in self.templates:
            print("Gerando copia de " + t.rawRepr)
            for a in angles:
                print("Angulo = " + str(a))
                rty = rface.getRotationMatrix(a, 'y')
                nobj = copy.deepcopy(t)
                pathBroken = nobj.rawRepr.split(os.path.sep)
                fileName = pathBroken[-1]
                pathBroken = os.path.sep.join(pathBroken[:-2])
                fileName = fileName.split('_')
                if (fileName[-1] == 'symmetricfilled.obj'):
                    fileName = '_'.join(fileName[0:-1])
                else:
                    fileName = '_'.join(fileName)[0:-4]
                pathBroken = os.path.join(pathBroken, 'Depth', 'DepthBMP', fileName + '_rotate_' + str(a) + '.obj')

                nobj.rawRepr = pathBroken
                nobj.faceMarks = rface.multiplyMatrices(nobj.faceMarks, rty)
                addTemplates.append(nobj)

        self.templates = self.templates + addTemplates

    def generateDatabaseFile(self, path, otherBases=None, outputDesired='SVMTorchFormat', excludeFiles=[]):
        currSearch = self.templates
        if otherBases:
            for o in otherBases:
                currSearch = currSearch + o.templates

        currFiles = []
        for f in currSearch:
            originalPathFile = f.rawRepr.split(os.path.sep)
            originalPathFile = originalPathFile[-1].split('_')[-1]
            if not (originalPathFile in excludeFiles):
                currFiles.append(f)

        return getattr(self, outputDesired)(currFiles, path)

    def SVMTorchFormat(self, data, path):
        f = open(path, 'w')
        f.write(str(len(data)) + ' ' + str(len(data[0].features) + 1) + '\n')
        for t in data:
            print("Imprimindo linha")
            f.write(' '.join([str(e) for e in t.features]) + ' ' + str(t.itemClass - 1) + '\n')
        f.close()
        return []

    def caffeFormat(self, data, path):
        f = open(path, 'w')
        for t in data:
            f.write(' '.join([str(e) for e in t.features]) + '\n')
        f.close()
        return []

    def h5Format(self, data, path):
        import h5py
        X = np.zeros((len(data), len(data[0].features)))
        Y = np.zeros((len(data), 1))
        for t in range(len(data)):
            X[t] = data[t].features
            Y[t] = data[t].itemClass

        with h5py.File(path, 'w') as H:
            H.create_dataset('X', data=X)
            H.create_dataset('Y', data=Y)
        with open(path[:-3] + '_list.txt', 'w') as L:
            L.write(path)

    def imageForCaffe(self, data, path):
        dataNum = 0
        for t in data:
            dataNum = dataNum + 1
            fixedFeature = np.array(t.features).reshape(int(len(t.features) / 14), 14)
            f = open(os.path.join(path, str(t.itemClass) + '_' + t.typeTemplate + '_' + str(dataNum) + '.txt'), 'w')
            for i in fixedFeature:
                finalPrint = ''
                for j in i:
                    finalPrint = finalPrint + str(j) + " "
                f.write(finalPrint + '\n')
            f.close()

        return []

    def generateCharsClasses(self, data, path):
        returnArray = [[], []]
        for t in data:
            returnArray[0].append(t.features)
            returnArray[1].append(t.itemClass)

        return returnArray