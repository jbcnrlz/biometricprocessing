import os, random, PIL.ImageOps
from baseClasses.Template import *
from helper.functions import outputObj, loadOBJ, scaleValues


class FRGCTemplate(Template):

    folderTemplate = None
    layersChar = None

    def __init__(self,pathFile,subject,dataset=None):
        self.itemClass = subject
        self.faceMarks = []
        super().__init__(pathFile,None,True,dataset)

    def loadImageData(self):
        self.imageLoaded = True
        if (self.rawRepr[-3:]=='obj'):
            a, b, imageFace, y = loadOBJ(self.rawRepr)
            self.image = np.array(imageFace)
        else:
            self.image = np.array(im.open(self.rawRepr).convert('L'))

    def loadImage(self):
        if self.rawRepr[-3:] == 'bmp':
            imageFace = im.open(self.rawRepr).convert('L')
            self.layersChar = np.zeros((imageFace.size[0],imageFace.size[1],4))
        else:
            a, b, imageFace, y = loadOBJ(os.path.join('temporaryTemplate',str(self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.obj'))
        self.image = imageFace

    def loadNewDepthImage(self):
        if self.lazyLoading:
            self.rawRepr = self.rawRepr[0:-4] + '_newdepth.jpeg'
        else:
            
            self.image= np.array(im.open(self.rawRepr[0:-4] + '_newdepth.jpeg'))

    def saveNewDepth(self):
        if self.rawRepr[-3:] == 'obj':
            scaleData = scaleValues(0,255,self.image.T)
            sImage = im.fromarray(scaleData).convert('RGB')
            sImage.save(self.rawRepr[0:-4]+'_newdepth.jpeg')
            invImage = PIL.ImageOps.invert(sImage)
            invImage.save(self.rawRepr[0:-4] + '_newdepth_inv.jpeg')
        else:
            sImage = im.fromarray(self.image.T).convert('RGB')
            sImage.save(self.rawRepr[0:-4] + '_newdepth.jpeg')

    def save(self,saveOnPath=False,prefix='_segmented'):
        if (not saveOnPath):
            if (not os.path.exists('temporaryTemplate')):
                os.makedirs('temporaryTemplate')

            outputObj(self.image,os.path.join('temporaryTemplate',str(self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.obj'))
            self.outputMarks()
        else:
            if (self.rawRepr[0:-4] == 'jpeg'):
                sImage = im.fromarray(self.image).convert('RGB')
                sImage.save(self.rawRepr[0:-4] + prefix + '.jpeg')
            else:
                outputObj(self.image,self.rawRepr[0:-4] + prefix + '.obj')

    def outputMarks(self,saveOnPath=False,typeTemplate='Depth'):
        if (not saveOnPath):
            if (not os.path.exists('temporaryTemplate')):
                os.makedirs('temporaryTemplate')
            f = open(os.path.join('temporaryTemplate',str(self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.txt'),'w')
            f.write('\n'.join([ '\t'.join(map(str,x)) for x in self.faceMarks]))
            f.close()
        else:
            filesPath = self.rawRepr.split('/')
            fileName = filesPath[-1].split('.')
            fileName = fileName[0].split('_')
            if typeTemplate == 'Depth':
                filesPath = os.path.join('/'.join(filesPath[:-3]),'Mark','MarkRGB','rgb_'+fileName[1]+'_'+filesPath[:-3][len(filesPath) - 4]+'_'+fileName[3]+'_Points_newdepth.txt')
            elif typeTemplate == '3DObj':
                filesPath = os.path.join('/'.join(filesPath[:-3]),'Mark','Mark3DObj','depth_'+fileName[1]+'_'+filesPath[:-3][len(filesPath) - 4]+'_'+fileName[3]+'_Points_OBJ_newdepth.txt')
            f = open(filesPath,'w')
            f.write('\n'.join([ '\t'.join(map(str,x)) for x in self.faceMarks]))
            f.close()

    def loadMarks(self):
        pathDistorted = self.rawRepr.split(os.path.sep)
        fileName = pathDistorted[-1].split('.')
        fileName = fileName[0] + '_fp2.txt'
        pathDistorted = pathDistorted[:-1]
        pathDistorted = os.path.sep.join(pathDistorted) + os.path.sep + fileName
        batata = open(pathDistorted,'r')
        for i in range(3):
            batata.readline()
            noseData = batata.readline()
            self.faceMarks.append(list(map(int,noseData.split(' '))))
        batata.close()

    def saveTXTChars(self):
        f = open('teste.txt','w')
        f.write(' '.join(map(str,self.features)) + '\n')
        f.close()

    def saveImageTraining(self,avgImageSave=True,pathImage='generated_images_lbp_frgc'):
        #imageSaveDLP = np.array(self.layersChar)        
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = im.fromarray(np.uint8(self.layersChar))
        pathNImage = pathImage+'/'+str(self.itemClass) + '_' + fullPath +'.png'
        while (os.path.exists(pathNImage)):
            idxRandomIm = random.randint(1,255)
            pathNImage = pathImage+'/'+str(self.itemClass) + '_' + fullPath +'_'+str(idxRandomIm)+'.png'
            
        imageSaveDLP.save(pathNImage)

    def saveHistogramImage(self,imageSave=None,folder='generated_images_wld'):
        if (imageSave is None):
            imageSave = self.features

        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = im.fromarray(imageSave)
        pathNImage = folder + '/'+str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath +'.jpg'
        while (os.path.exists(pathNImage)):
            idxRandomIm = random.randint(1,255)
            pathNImage = folder+'/'+str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath +'_'+str(idxRandomIm)+'.png'

        imageSaveDLP.convert('RGB').save(pathNImage)