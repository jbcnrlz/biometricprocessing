from math import sqrt
from baseClasses.Template import *
from helper.functions import outputObj, loadOBJ
from PIL import Image as im
import os, numpy as np, random, pcl, math

class LFWTemplate(Template):

    folderTemplate = None
    faceMarks = []
    layersChar = None

    def __init__(self,pathFile,subject,lazyData=False,dataset=None):
        self.itemClass = subject
        self.nFacets = []
        self.facets = []
        self.normals = []
        self.facetFunctionData = []
        super().__init__(pathFile,None,lazyData,dataset)

    def save(self,saveOnPath=False):
        if (not saveOnPath):
            if (not os.path.exists('temporaryTemplate')):
                os.makedirs('temporaryTemplate')

            outputObj(self.image,os.path.join('temporaryTemplate',str(self.itemClass) + '_' + self.folderTemplate + '_' + self.typeTemplate + '.obj'))
            self.outputMarks()
        else:
            self.image.save(self.rawRepr[0:-4] + '_newdepth.bmp')


    def loadMarks(self,typeTemplate='Depth'):
        self.faceMarks = []

    def facetFunction(self):
        normalAngAvg = []
        for f in range(len(self.facets)):
            currNormal = self.normals[f]
            angAvg = 0
            for i in range(len(self.nFacets[f])):
                na = np.linalg.norm(currNormal)
                nb = np.linalg.norm(self.normals[self.nFacets[f][i]])
                ang= np.dot(currNormal,self.normals[self.nFacets[f][i]]) / (na*nb)
                angAvg += math.acos(ang)

            angAvg = angAvg / len(self.nFacets[f]) if len(self.nFacets[f]) > 0 else 0
            normalAngAvg.append(angAvg)

        self.facetFunctionData = normalAngAvg

    def calculateNormal(self,p1, p2, p3):
        V = p2 - p1
        W = p3 - p1
        nx = (V[1] * W[2]) - (V[2] * W[1])
        ny = (V[2] * W[0]) - (V[0] * W[2])
        nz = (V[0] * W[1]) - (V[1] * W[0])
        sumN = nx + ny + nz
        return np.array([nx, ny, nz]) / sumN

    def generateNeighbours(self):
        self.nFacets = [[] for i in range(len(self.facets))]
        #self.nFacets = []
        for i in range(len(self.facets)):
            print(i)
            if len(self.nFacets[i]) == 3:
                print('pulou')
                continue

            neighs = [k for k in range(len(self.facets)) if len(set(self.facets[i]) & set(self.facets[k])) == 2]
            for j in neighs:
                if i not in self.nFacets[j]:
                    self.nFacets[j].append(i)
            self.nFacets[i] = neighs
            '''
            if len(self.nFacets[i]) == 3:
                continue

            for j in range(len(self.facets)):
                if len(set(self.facets[i]) & set(self.facets[j])) == 2:
                    self.nFacets[i].append(j)
                    self.nFacets[j].append(i)

                if len(self.nFacets[i]) == 3:
                    break
            '''
    def loadNewDepthImage(self):
        self.image = im.open(self.rawRepr[0:-4] + '_newdepth.bmp')
        self.loadMarks('newdepth')

    def calculateNormal(self,p1, p2, p3):
        V = p2 - p1
        W = p3 - p1
        nx = (V[1] * W[2]) - (V[2] * W[1])
        ny = (V[2] * W[0]) - (V[0] * W[2])
        nz = (V[0] * W[1]) - (V[1] * W[0])
        sumN = nx + ny + nz
        return np.array([nx, ny, nz]) / sumN

    def saveImageTraining(self,avgImageSave=True,pathImage='generated_images_lbp'):
        #imageSaveDLP = np.array(self.layersChar)
        if (avgImageSave):
            avImage = np.zeros((self.layersChar.shape[0],self.layersChar.shape[1]))        
            for i in range(self.layersChar.shape[0]):
                for j in range(self.layersChar.shape[1]):
                    avImage[i,j] = self.layersChar[i,j,0] + self.layersChar[i,j,1] + self.layersChar[i,j,2] + self.layersChar[i,j,3]
                    avImage[i,j] = avImage[i,j] / 4
            avImage = im.fromarray(np.uint8(avImage))
            #avImage.save(pathImage+'/averageImage/'+str(self.itemClass) + '_' + self.folderTemplate + '_' + fullPath +'.jpg')
        
        fullPath = self.rawRepr.split(os.path.sep)
        fullPath = fullPath[-1].split('.')
        fullPath = fullPath[0]
        imageSaveDLP = im.fromarray(np.uint8(self.layersChar))
        pathNImage = pathImage+'/'+str(self.itemClass) + '_' + fullPath +'.png'
        while (os.path.exists(pathNImage)):
            idxRandomIm = random.randint(1,255)
            pathNImage = pathImage+'/'+str(self.itemClass) + '_' + fullPath +'_'+str(idxRandomIm)+'.png'
            
        print("Gerando imagem de "+pathNImage)
        imageSaveDLP.save(pathNImage)
        return pathNImage

    def generateNormals(self):
        print('Iniciou normais')
        if not self.imageLoaded:
            self.loadImageData()
        if type(self.image) is not np.ndarray:
            self.image = np.array(self.image)

        for f in self.facets:
            nf = self.calculateNormal(self.image[f[0]],self.image[f[1]],self.image[f[2]])
            self.normals.append(nf)
        print('terminou normais')
        print('iniciou vizinhos')
        self.generateNeighbours()
        print('terminou vizinhos')
        print('iniciou descriptor')
        self.calculateDescriptorFacet()
        print('terminou descriptor')
        self.nFacets = None
        self.normals = None
        self.image = None
        self.imageLoaded = False

    def calculateDescriptorFacet(self):
        for f in range(len(self.facets)):
            currNormal = self.normals[f]
            angAvg = 0
            for i in range(len(self.nFacets[f])):
                na = np.linalg.norm(currNormal)
                nb = np.linalg.norm(self.normals[self.nFacets[f][i]])
                ang = np.dot(currNormal, self.normals[self.nFacets[f][i]]) / (na * nb)
                angAvg += math.acos(ang)

            angAvg = angAvg / len(self.nFacets[f]) if len(self.nFacets[f]) > 0 else 0
            self.facetFunctionData.append(angAvg)