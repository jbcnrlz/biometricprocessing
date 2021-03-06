import os, random, PIL.ImageOps, h5py
from baseClasses.Template import *
from helper.functions import outputObj, loadOBJ, scaleValues
from PIL import Image as im

class BUTemplate(Template):
    folderTemplate = None
    layersChar = None

    def __init__(self, pathFile, subject, dataset=None):
        self.itemClass = subject
        self.faceMarks = []
        super().__init__(pathFile, None, True, dataset)

    def loadLandmarksfromFile(self):
        returnLandMarks = {}
        with open(self.rawRepr[:-3]+'bnd','r') as lFile:
            fullLines = lFile.readlines()
            for i, l in enumerate(fullLines[3:]):
                fixed = l.replace('\t\t',' ')
                fixed = fixed.replace('\t',' ').strip('\n').split()
                fixed = list(map(float,fixed))
                returnLandMarks[len(returnLandMarks)] = fixed[1:]

        return returnLandMarks

    def loadImageData(self):
        self.imageLoaded = True
        if (self.rawRepr[-3:] == 'mat'):
            arrays = {}
            try:
                with h5py.File(self.rawRepr) as fPy:
                    for k, v in fPy.items():
                        arrays[k] = np.array(v)
                self.image = arrays['vertex'].T
                self.faceMarks = arrays['lm3d'].T
            except:
                import scipy.io as sio
                arrays = sio.loadmat(self.rawRepr)
                self.image = arrays['d'][:,0:3]
                self.faceMarks = self.loadLandmarksfromFile()
        elif (self.rawRepr[-4:] == 'jpeg') or (self.rawRepr[-3:] == 'bmp') or (self.rawRepr[-3:] == 'jpg'):
            self.image = np.array(im.open(self.rawRepr).convert('L'))
        else:
            holder = []
            with open(self.rawRepr, "rb") as vrml:
                vrmlFileContent = vrml.readlines()
                quantity = vrmlFileContent[1].decode().split(' ')
                quantity = int(quantity[0])
                for point in vrmlFileContent[2:quantity+2]:
                    holder.append(list(map(float,point.decode().split(' '))))

            self.image = np.array(holder)
            self.faceMarks = self.loadLandmarksfromFile()


    def loadImage(self):
        if self.rawRepr[-3:] == 'bmp':
            imageFace = im.open(self.rawRepr).convert('L')
            self.layersChar = np.zeros((imageFace.size[0], imageFace.size[1], 4))
        elif (self.rawRepr[-3:] == 'mat'):
            arrays = {}
            with h5py.File(self.rawRepr) as fPy:
                for k, v in fPy.items():
                    arrays[k] = np.array(v)
            self.image = np.array(arrays['vertex'])
            self.faceMarks = arrays['lm3d']
        else:
            a, b, imageFace, y = loadOBJ(os.path.join('temporaryTemplate', str(
                self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.obj'))
        self.image = imageFace

    def loadNewDepthImage(self):
        if self.lazyLoading:
            self.rawRepr = self.rawRepr[0:-4] + '_newdepth.jpeg'
        else:

            self.image = np.array(im.open(self.rawRepr[0:-4] + '_newdepth.jpeg'))

    def saveNewDepth(self):
        sImage = im.fromarray(self.image).convert('RGB')
        sImage = sImage.rotate(180)
        sImage.save(self.rawRepr[0:-4] + '_newdepth.jpeg')

    def save(self, saveOnPath=False, prefix='_segmented'):
        if (not saveOnPath):
            if (not os.path.exists('temporaryTemplate')):
                os.makedirs('temporaryTemplate')

            outputObj(self.image, os.path.join('temporaryTemplate', str(
                self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.obj'))
            self.outputMarks()
        else:
            if (self.rawRepr[0:-4] == 'jpeg'):
                sImage = im.fromarray(self.image).convert('RGB')
                sImage.save(self.rawRepr[0:-4] + prefix + '.jpeg')
            elif (type(self.image) is PIL.Image.Image):
                self.image.save(self.rawRepr[0:-4] + prefix + '.jpeg')
            else:
                outputObj(self.image, self.rawRepr[0:-4] + prefix + '.obj')

    def outputMarks(self, saveOnPath=False, typeTemplate='Depth'):
        if (not saveOnPath):
            if (not os.path.exists('temporaryTemplate')):
                os.makedirs('temporaryTemplate')
            f = open(os.path.join('temporaryTemplate',
                                  str(self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.txt'),
                     'w')
            f.write('\n'.join(['\t'.join(map(str, x)) for x in self.faceMarks]))
            f.close()
        else:
            filesPath = self.rawRepr.split('/')
            fileName = filesPath[-1].split('.')
            fileName = fileName[0].split('_')
            if typeTemplate == 'Depth':
                filesPath = os.path.join('/'.join(filesPath[:-3]), 'Mark', 'MarkRGB',
                                         'rgb_' + fileName[1] + '_' + filesPath[:-3][len(filesPath) - 4] + '_' +
                                         fileName[3] + '_Points_newdepth.txt')
            elif typeTemplate == '3DObj':
                filesPath = os.path.join('/'.join(filesPath[:-3]), 'Mark', 'Mark3DObj',
                                         'depth_' + fileName[1] + '_' + filesPath[:-3][len(filesPath) - 4] + '_' +
                                         fileName[3] + '_Points_OBJ_newdepth.txt')
            f = open(filesPath, 'w')
            f.write('\n'.join(['\t'.join(map(str, x)) for x in self.faceMarks]))
            f.close()

    def loadMarks(self):
        if self.faceMarks is None:
            pathDistorted = self.rawRepr.split(os.path.sep)
            fileName = pathDistorted[-1].split('.')
            fileName = fileName[0] + '_fp2.txt'
            pathDistorted = pathDistorted[:-1]
            pathDistorted = os.path.sep.join(pathDistorted) + os.path.sep + fileName
            batata = open(pathDistorted, 'r')
            for i in range(3):
                batata.readline()
                noseData = batata.readline()
                self.faceMarks.append(list(map(int, noseData.split(' '))))
            batata.close()

    def saveTXTChars(self):
        f = open('teste.txt', 'w')
        f.write(' '.join(map(str, self.features)) + '\n')
        f.close()

    def isFileExists(self,pathImage):
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        pathNImage = pathImage + '/' + str(self.itemClass) + '_' + fullPath + '.png'
        return os.path.exists(pathNImage)

    def saveImageTraining(self, avgImageSave=True, pathImage='generated_images_lbp_frgc'):
        if (not os.path.exists(pathImage)):
            os.makedirs(pathImage)
        # imageSaveDLP = np.array(self.layersChar)
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = im.fromarray(np.uint8(self.layersChar))
        pathNImage = pathImage + '/' + str(self.itemClass) + '_' + fullPath + '.png'
        imageSaveDLP.save(pathNImage)