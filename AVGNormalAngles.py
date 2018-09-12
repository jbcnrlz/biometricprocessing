import numpy as np, math, operator, os, logging, scipy.ndimage as ndimage, matlab.engine
from baseClasses.BiometricProcessing import *
from helper.functions import generateHistogram, loadOBJ, generateHistogramUniform, bilinear_interpolation, mergeArraysDiff, minmax, printProgressBar, outputObj
from PIL import Image as im
from scipy import interpolate
from scipy.spatial.distance import *
from scipy.special import expit
from mahotas.interpolate import shift
from math import atan2,degrees, acos, sqrt
from datetime import datetime

class AVGNormalAngles(BiometricProcessing):

    def __init__(self,windowsize,binsize,database):
        self.windowSize = windowsize
        self.binsize = binsize
        self.databases = database

    def getLimits(self,fullFace):
        smallerValues = [float('inf'),float('inf')]
        biggerValues = [-float('inf'),-float('inf')]
        for f in fullFace:
            if (f[0] < smallerValues[0]):
                smallerValues[0] = f[0]

            if (f[1] < smallerValues[1]):
                smallerValues[1] = f[1]

            if (f[0] > biggerValues[0]):
                biggerValues[0] = f[0]

            if (f[1] > biggerValues[1]):
                biggerValues[1] = f[1]

        return smallerValues + biggerValues

    def arrangeListByDistance(self,center,listN,pointsQ=None):
        listWithDistance = [(l.tolist(),euclidean(center,l)) for l in listN]
        dTypeList = [('point',list),('distance',float)]
        listWithDistance = np.sort(np.array(listWithDistance,dtype=dTypeList),order='distance')
        if (not pointsQ):
            return np.array(listWithDistance['point'].tolist())
        else:
            return np.array(listWithDistance['point'].tolist()[:pointsQ])

    def calculateNormal(self,p1,p2,center):
        v1 = center - p1
        v2 = center - p2
        return np.cross(v1,v2)

    def findAngleBetweenVectors(self,v1,v2):
        if (type(v1) is not np.ndarray):
            v1 = np.array(v1)
        if (type(v2) is not np.ndarray):
            v2 = np.array(v2)

        m1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        m2 = sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        dot = np.dot(v1,v2)
        try:
            return acos(dot / (m1*m2))
        except:
            return 0

    def generateWindows(self,fullFace,limits,size=(100,100)):
        windowXSize = 0
        if (limits[0] < 0):
            windowXSize = (limits[2] - limits[0]) / size[0]
        else:
            windowXSize = (limits[2] + limits[0]) / size[0]

        windowYSize = 0
        if (limits[1] < 0):
            windowYSize = (limits[3] - limits[1]) / size[1]
        else:
            windowYSize = (limits[3] + limits[1]) / size[1]

        currentY = limits[1]
        region = []
        while(currentY <= limits[3]):
            currentX = limits[0]
            while(currentX <= limits[2]):
                region.append([])
                for f in fullFace:
                    if (f[0] >= currentX) and (f[0] < currentX + windowXSize) and (f[1] >= currentY) and (f[1] < currentY + windowYSize):
                        region[len(region) - 1].append(f)

                currentX += windowXSize
            currentY += windowYSize

        return region

    def setupTemplate(self,template):
        return template

    def cleanupTemplate(self,template):
        return template

    def saveTXTMatlab(self,path,face):
        f = open(path,'w')
        for p in face:
            f.write(' '.join(map(str,p)) + '\n')
        f.close()

    '''
    def extractKH(self,x,y,z):
        if (type(x) is not np.ndarray):
            x = np.array(x)

        if (type(y) is not np.ndarray):
            y = np.array(y)

        if (type(z) is not np.ndarray):
            z = np.array(z)

        Xu, Xv = np.gradient(x)
        Yu, Yv = np.gradient(y)
        Zu, Zv = np.gradient(z)

        Xuu, Xuv = np.gradient(Xu)
        Yuu, Yuv = np.gradient(Yu)
        Zuu, Zuv = np.gradient(Zu)

        Xuv, Xvv = np.gradient(Xv)
        Yuv, Yvv = np.gradient(Yv)
        Zuv, Zvv = np.gradient(Zv)

        Xu = np.array([Xu.flatten(), Yu.flatten(), Zu.flatten()]).T
        Xv = np.array([Xv.flatten(), Yv.flatten(), Zv.flatten()]).T
        Xuu = np.array([Xuu.flatten(), Yuu.flatten(), Zuu.flatten()]).T
        Xuv = np.array([Xuv.flatten(), Yuv.flatten(), Zuv.flatten()]).T
        Xvv = np.array([Xvv.flatten(), Yvv.flatten(), Zvv.flatten()]).T

        E = np.array([np.dot(i,i) for i in Xu])
        F = np.array([np.dot(Xu[i],Xv[i]) for i in range(Xu.shape[0])])
        G = np.array([np.dot(i,i) for i in Xv])

        m = np.cross(Xu,Xv)
        p = np.sqrt([np.dot(i,i) for i in m])
        n = np.array(m / np.array([p,p,p]).T)

        L = np.array([np.dot(Xuu[i],n[i]) for i in range(Xuu.shape[0])])
        M = np.array([np.dot(Xuv[i],n[i]) for i in range(Xuv.shape[0])])
        N = np.array([np.dot(Xvv[i],n[i]) for i in range(Xvv.shape[0])])

        K = ((L*N) - (M**2)) / ((E*G) - (F**2))
        K = K.reshape(z.shape[0],z.shape[1])

        H = -((E*N) + (G*L) - (F*M*2)) / (2*((E*G) - (F**2)))
        H = H.reshape(z.shape[0],z.shape[1])

        return K,H
    '''

    def surface_curvature(self,X,Y,Z):
        if (type(X) is not np.ndarray):
            X = np.array(X)

        if (type(Y) is not np.ndarray):
            Y = np.array(Y)

        if (type(Z) is not np.ndarray):
            Z = np.array(Z)



        (lr,lb)=X.shape

        #First Derivatives
        Xv,Xu=np.gradient(X)
        Yv,Yu=np.gradient(Y)
        Zv,Zu=np.gradient(Z)

        #Second Derivatives
        Xuv,Xuu=np.gradient(Xu)
        Yuv,Yuu=np.gradient(Yu)
        Zuv,Zuu=np.gradient(Zu)   

        Xvv,Xuv=np.gradient(Xv)
        Yvv,Yuv=np.gradient(Yv)
        Zvv,Zuv=np.gradient(Zv) 

        #2D to 1D conversion 
        #Reshape to 1D vectors
        Xu=np.reshape(Xu,lr*lb)
        Yu=np.reshape(Yu,lr*lb)
        Zu=np.reshape(Zu,lr*lb)
        Xv=np.reshape(Xv,lr*lb)
        Yv=np.reshape(Yv,lr*lb)
        Zv=np.reshape(Zv,lr*lb)
        Xuu=np.reshape(Xuu,lr*lb)
        Yuu=np.reshape(Yuu,lr*lb)
        Zuu=np.reshape(Zuu,lr*lb)
        Xuv=np.reshape(Xuv,lr*lb)
        Yuv=np.reshape(Yuv,lr*lb)
        Zuv=np.reshape(Zuv,lr*lb)
        Xvv=np.reshape(Xvv,lr*lb)
        Yvv=np.reshape(Yvv,lr*lb)
        Zvv=np.reshape(Zvv,lr*lb)

        Xu=np.c_[Xu, Yu, Zu]
        Xv=np.c_[Xv, Yv, Zv]
        Xuu=np.c_[Xuu, Yuu, Zuu]
        Xuv=np.c_[Xuv, Yuv, Zuv]
        Xvv=np.c_[Xvv, Yvv, Zvv]

        #% First fundamental Coeffecients of the surface (E,F,G)
        E=np.einsum('ij,ij->i', Xu, Xu) 
        F=np.einsum('ij,ij->i', Xu, Xv) 
        G=np.einsum('ij,ij->i', Xv, Xv) 

        m=np.cross(Xu,Xv,axisa=1, axisb=1) 
        p=np.sqrt(np.einsum('ij,ij->i', m, m)) 
        n=m/np.c_[p,p,p]
        # n is the normal
        #% Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)
        L= np.einsum('ij,ij->i', Xuu, n) #e
        M= np.einsum('ij,ij->i', Xuv, n) #f
        N= np.einsum('ij,ij->i', Xvv, n) #g

        # Alternative formula for gaussian curvature in wiki 
        # K = det(second fundamental) / det(first fundamental)
        #% Gaussian Curvature
        K=(L*N-M**2)/(E*G-L**2)
        K=np.reshape(K,lr*lb)
        #wiki trace of (second fundamental)(first fundamental inverse)
        #% Mean Curvature
        H = (E*N + G*L - 2*F*M)/((E*G - F**2))
        H = np.reshape(H,lr*lb)
        
        #% Principle Curvatures
        Pmax = H + np.sqrt(H**2 - K)
        Pmin = H - np.sqrt(H**2 - K)
        #[Pmax, Pmin]
        Principle = [Pmax,Pmin]
        return Principle

    def generateDepthMap(self,cpoint,eng=None):
        if eng == None:
            eng = matlab.engine.start_matlab()

        cpoint = matlab.double(cpoint)        
        x,y,z= eng.generateNewDepthMapsXYZ(cpoint,nargout=3)
        return x,y,z

    def extractKH_matlab(self,path,w,h):
        eng = matlab.engine.start_matlab()
        k,h= eng.extractChars(path,nargout=2)
        return k, h

    
    def findNeighours(self,data,neighbour):
        distances = []
        for d in data:
            distances.append(euclidean(d[0:-1],neighbour))

        return sorted(range(len(distances)), key=lambda k: distances[k])    

    def generateGrid(self,data):
        '''
        if (type(data) is not np.ndarray):
            data = np.array(data)
    
        size = [i for i in range(math.ceil(sqrt(len(data))))]
        f = interpolate.interp2d(data[0:,0],data[0:,1],data[0:,2], kind='cubic')
        xnew = np.arange(data[0:,0].min(),data[0:,0].max(),np.std(data[0:,0]))
        ynew = np.arange(data[0:,1].min(),data[0:,1].max(),np.std(data[0:,1]))
        return f(xnew,ynew)

        '''
        if (type(data) is not np.ndarray):
            data = np.array(data)
        size = math.ceil(sqrt(len(data)))
        grid = np.zeros((size,size))

        for i in range(size):
            for j in range(size):
                neighsIdx = self.findNeighours(data,[i,j])
                grid[i,j] = np.mean(data[0:,2])

        return grid

        '''
        print(data)
        input()

        
        for i in range(size):
            for j in range(size):
                smalDist = math.inf
                idxSmall = None
                for d in range(data.shape[0]):
                    dist = euclidean([i,j],data[d][0:-1])
                    if dist < smalDist:
                        smalDist = dist
                        idxSmall = d

                grid[i,j] = data[idxSmall][-1]
                data = np.delete(data,idxSmall,0)

        print(grid)
        input()
        return grid
        '''

    def featureExtraction(self,excludeFile=[]):
        print('============= Iniciando Feature Extraction =============')
        eng = matlab.engine.start_matlab()
        for database in self.databases:
            for template in database.templates:
                template.features = [[],[]]
                inicio = datetime.now()
                print('Gerando descritor de: '+str(template.itemClass))
                limitValues = self.getLimits(template.image)
                windowsGenerated = self.generateWindows(template.image,limitValues,self.windowSize)
                step = 0
                printProgressBar(step,len(windowsGenerated),prefix='Progresso:',suffix='Completo',length=50)
                for w in windowsGenerated:
                    step += 1
                    if (len(w) <= 10):
                        currHist = [0.0] * self.binsize
                        printProgressBar(step,len(windowsGenerated),prefix='Progresso:',suffix='Completo',length=50)
                        template.features[0] = template.features[0] + currHist
                        template.features[1] = template.features[1] + currHist
                        continue

                    x,y,z = self.generateDepthMap(w,eng)
                    k,h = np.array(self.surface_curvature(x,y,z))
                    #k,h = self.extractKH(x,y,z)
                    print(k)
                    k = np.histogram(k,bins=self.binsize)[0]
                    template.features[0] = template.features[0] + k.tolist()
                    h = np.histogram(h,bins=self.binsize)[0]
                    template.features[1] = template.features[1] + h.tolist()

                template.features = template.features[0] + template.features[1]
                fim = datetime.now() - inicio
                print("Extração levou %d segundos" % fim.seconds)
                '''
                template3Dobj = template.rawRepr.split(os.path.sep)[:-2]
                folderType = template3Dobj[template3Dobj.index(subject) + 1]
                txtFilePath = os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'_processing_matlab.obj')
                self.saveTXTMatlab(txtFilePath,w)
                k,h = self.extractKH(txtFilePath,10,10)
                os.remove(txtFilePath)
                k = np.histogram(k,density=True)[0]
                template.features[0] = template.features[0] + k.tolist()
                h = np.histogram(h,density=True)[0]
                template.features[1] = template.features[1] + h.tolist()
                printProgressBar(step,len(windowsGenerated),prefix='Progresso:',suffix='Completo',length=50)
                '''
    '''
    def featureExtraction(self,excludeFile=[]):
        for database in self.databases:
            for template in database.templates:
                print('Gerando descritor de: '+str(template.itemClass))
                limitValues = self.getLimits(template.image)
                windowsGenerated = self.generateWindows(template.image,limitValues,self.windowSize)
                print(windowsGenerated)
                print(len(windowsGenerated))
                descriptor = [0.0]  * (self.windowSize[0] * self.windowSize[1] * self.binsize)
                currWindow = 0
                for w in windowsGenerated:
                    if (len(w) <= 10):
                        continue

                    print(len(w))
                    input()

                    self.saveTXTMatlab('currentfile.txt',w)
                    self.extractKH('currentfile.txt',10,10)
                    os.remove('currentfile.txt')

                    pointsRegion = np.array(w)
                    histoPoints = []
                    for r in range(pointsRegion.shape[0]):
                        avgRegion = []
                        cpoint = pointsRegion[r]
                        neighbors = self.arrangeListByDistance(cpoint,pointsRegion,8)
                        for cidx in range(neighbors.shape[0]):
                            nidx = cidx + 1 if cidx < neighbors.shape[0] -1 else 0
                            normalN = self.calculateNormal(neighbors[cidx],neighbors[nidx],cpoint)
                            v1 = neighbors[cidx] - cpoint
                            v2 = normalN - cpoint
                            angle = self.findAngleBetweenVectors(v1,v2)
                            if (not math.isnan(angle)):
                                avgRegion.append(angle)

                        if (len(avgRegion) > 0):
                            diff = 0
                            for a in avgRegion:
                                diff = a - diff
                                
                            avgRegion = np.arctan(diff)
                            histoPoints.append(avgRegion)
                        else:
                            histoPoints.append(0)

                    if (len(histoPoints) > 0):
                        histoPoints, bins = np.histogram(np.array(histoPoints),density=True,bins=self.binsize)
                        descriptor[self.binsize * currWindow:self.binsize * (currWindow + 1)] = histoPoints.tolist()

                    currWindow += 1

                template.features = descriptor
                print(len(template.features))
    '''