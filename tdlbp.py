import math, operator
from scipy.special import expit
from sklearn.preprocessing import normalize
from FRGCTemplate import *
from LFWTemplate import *
from BosphorusTemplate import *
from baseClasses.BiometricProcessing import *
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp2d
from helper.functions import generateHistogram, generateHistogramUniform, generateArrayUniform, zFunc, fitPlane


class ThreeDLBP(BiometricProcessing):

    def __init__(self, windowsize, binsize, database,generateImages=True):
        self.windowSize = windowsize
        self.binsize = binsize
        self.databases = database
        self.methodName = '3DLBP'
        self.generateImagesTrainig = generateImages
        self.sigmoid_table = {}
        super().__init__()

    def saveDebug(self, folder, data):
        with open(folder + '/subtraction.txt', 'a') as rty:
            rty.write(str(data) + '\n')

    def generateCode(self, image, center, typeOp='Normal',truncMaskPlus=None,truncMaskMinus=None,firstLayer='lbp'):
        idxs = [
            (center[0] - 1, center[1] - 1),
            (center[0] - 1, center[1]),
            (center[0] - 1, center[1] + 1),
            (center[0], center[1] + 1),
            (center[0] + 1, center[1] + 1),
            (center[0] + 1, center[1]),
            (center[0] + 1, center[1] - 1),
            (center[0], center[1] - 1)
        ]
        layers = [[], [], [], []]
        points = []
        for i in idxs:
            subraction = 0
            if typeOp == 'Normal':
                try:
                    points.append([i[0], i[1], image[i[0]][i[1]]])
                    subraction = int(round(image[i[0]][i[1]] - image[center[0]][center[1]]))
                except:
                    print(image[i[0]][i[1]])
                    print(image[center[0]][center[1]])
                    print(subraction)
                if subraction < -7:
                    if truncMaskMinus is not None:
                        truncMaskMinus[i[0]][i[1]] += abs(subraction + 7)
                    subraction = -7
                elif subraction > 7:
                    if truncMaskPlus is not None:
                        truncMaskPlus[i[0]][i[1]] += subraction - 7
                    subraction = 7
            else:
                subraction = image[i[0]][i[1]] - image[center[0]][center[1]]
                signSub = -1 if subraction < 0 else 1
                subraction = np.histogram(expit(subraction), bins=7, range=[0, 1])[0]
                subraction = np.argwhere(subraction == 1)[0][0] * signSub
            bin = '{0:03b}'.format(abs(subraction))
            layers[0].append(str(int(subraction >= 0)))
            layers[1].append(bin[0])
            layers[2].append(bin[1])
            layers[3].append(bin[2])
        for l in range(len(layers)):
            layers[l] = int(''.join(layers[l]), 2)

        if firstLayer == 'angle':
            angle = self.getAnglePlaneAxis(np.array(points))
            layers[0] = np.histogram(angle, bins=255, range=[0, 2*np.pi])[0]

        return layers

    '''
    Gera indices quando usando P e R no 3DLBP ao inves da vizinhanca fixa de 3x3
    '''
    def generateImagePoints(self, image, centerIndex):
        pointsImage = []
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                pointsImage.append([y-centerIndex[1],(x*-1)+centerIndex[0],image[x,y]])
        return np.array(pointsImage)

    def getByPosition(self,x,y,image):
        for i in image:
            if i[0] == x and i[1] == y:
                return i

        else:
            return None

    def get_pixel2d(self,image,rows,cols,r,c,cval):
        if (r<0) or (r>=rows) or (c<0) or (c >= cols):
            return cval
        else:
            return image[r][c]

    def getAnglePlaneAxis(self,x):
        if type(x) is not np.ndarray:
            x = np.array(x)

        c, z = fitPlane(x)
        normals = []
        for i in range(len(x) - 2):
            v1 = x[-1] - np.array(x[i,:2].tolist() + [z[i]])
            v2 = x[-1] - np.array(x[i+1,:2].tolist() + [z[i+1]])
            normals.append(np.cross(v1,v2))

        finalNormal = np.mean(np.array(normals),axis=0)
        cosAngle = np.dot(finalNormal,np.array([0,0,1]))
        return np.degrees(np.arccos(cosAngle))

    def generateCodePR(self, image, center, xPositions, yPositions, type='Normal',truncMaskPlus=None,truncMaskMinus=None,firstLayer='lbp'):
        image = np.ascontiguousarray(image, dtype=np.double)
        xPositions = xPositions + center[0]
        yPositions = yPositions + center[1]
        idxs = np.array([x for x in zip(xPositions,yPositions)])
        idxs = idxs[-3:].tolist() + idxs[0:-3].tolist()
        points = []
        layers = [[], [], [], []]
        for i in idxs:
            subraction = 0
            if (not i[0].is_integer() or not i[1].is_integer()):
                xidxs = [math.floor(i[0]), math.ceil(i[0])]
                yidxs = [math.floor(i[1]), math.ceil(i[1])]

                imageDataBil = np.array([
                    (xidxs[0], yidxs[0], image[xidxs[0]][yidxs[0]]),
                    (xidxs[0], yidxs[1], image[xidxs[0]][yidxs[1]]),
                    (xidxs[1], yidxs[0], image[xidxs[1]][yidxs[0]]),
                    (xidxs[1], yidxs[1], image[xidxs[1]][yidxs[1]])
                ])
                nFunc = interp2d(imageDataBil[:,0],imageDataBil[:,1],imageDataBil[:,2])
                nv = nFunc(i[0], i[1])[0]
                points.append([i[0], i[1], nv])
                subraction = nv - image[center[0]][center[1]]
            else:
                points.append([i[0], i[1], image[int(i[0])][int(i[1])]])
                subraction = image[int(i[0])][int(i[1])] - image[center[0]][center[1]]

            if type == 'Normal':

                if subraction < -7:
                    if truncMaskMinus is not None and i[0].is_integer() and i[1].is_integer():
                        truncMaskMinus[int(i[0])][int(i[1])] += abs(subraction + 7)
                    elif truncMaskMinus is not None:
                        xidxs = [math.floor(i[0]), math.ceil(i[0])]
                        yidxs = [math.floor(i[1]), math.ceil(i[1])]
                        distancesEachItem = [euclidean([xidxs[interpX], yidxs[interpY]],i) for interpX in range(2) for interpY in range(2)]
                        distancesEachItem = normalize(np.array(distancesEachItem)[:,np.newaxis], axis=0).ravel()
                        place=0
                        for interpX in range(2):
                            for interpY in range(2):
                                truncMaskMinus[xidxs[interpX]][yidxs[interpY]] += abs(subraction + 7) * distancesEachItem[place]
                                place += 1

                    subraction = -7
                elif subraction > 7:
                    if truncMaskPlus is not None and i[0].is_integer() and i[1].is_integer():
                        truncMaskPlus[int(i[0])][int(i[1])] += abs(subraction - 7)
                    elif truncMaskPlus is not None:
                        xidxs = [math.floor(i[0]), math.ceil(i[0])]
                        yidxs = [math.floor(i[1]), math.ceil(i[1])]
                        distancesEachItem = [euclidean([xidxs[interpX], yidxs[interpY]],i) for interpX in range(2) for interpY in range(2)]
                        distancesEachItem = normalize(np.array(distancesEachItem)[:,np.newaxis], axis=0).ravel()
                        place=0
                        for interpX in range(2):
                            for interpY in range(2):
                                truncMaskPlus[xidxs[interpX]][yidxs[interpY]] += abs(subraction - 7) * distancesEachItem[place]
                                place += 1
                    subraction = 7
            elif type == 'wFunction':
                signSub = -1 if subraction < 0 else 1
                if subraction in self.sigmoid_table.keys():
                    subraction = self.sigmoid_table[subraction]
                else:
                    keyTable = subraction
                    subraction = zFunc(subraction,0.222)
                    self.sigmoid_table[keyTable] = subraction

                subraction = 0 if subraction < 0 else 1 if subraction > 1 else subraction
                subraction = np.histogram(subraction, bins=8, range=[0, 1])[0]
                subraction = (np.argwhere(subraction == 1)[0][0]+1) * signSub
            elif type == 'sigmoid':
                signSub = -1 if subraction < 0 else 1
                subraction = np.histogram(expit(subraction), bins=8, range=[0, 1])[0]
                subraction = np.argwhere(subraction == 1)[0][0] * signSub

            if firstLayer != 'angle':
                layers[0].append(str(int(subraction >= 0)))

                bin = '{0:03b}'.format(abs(int(round(subraction))))
                layers[1].append(bin[0])
                layers[2].append(bin[1])
                layers[3].append(bin[2])
            else:
                bin = '{0:03b}'.format(abs(int(round(subraction))))
                layers[0].append(bin[0])
                layers[1].append(bin[1])
                layers[2].append(bin[2])

            for l in range(len(layers)):
                if len(layers[l]) > 0:
                    layers[l] = int(''.join(layers[l]), 2)

            if firstLayer == 'angle':
                angle = round(self.getAnglePlaneAxis(np.array(points + [[center[0],center[1],image[center[0]][center[1]]]])))
                layers[3] = angle

        return layers

    def generateImageDescriptor(self, image, p=8, r=1, typeLBP='original', typeMeasurement='Normal',template=None,masks=False,firstLayer='lbp'):
        returnValue = [[], [], [], []]
        if masks:
            template.underFlow = np.zeros(image.shape)
            template.overFlow  = np.zeros(image.shape)
        else:
            template.underFlow = None
            template.overFlow  = None

        if typeLBP == 'pr':
            xPositions = np.round(- r * np.sin(2 * np.pi * np.arange(p, dtype=np.double) / p),5)
            yPositions = np.round(r * np.cos(2 * np.pi * np.arange(p, dtype=np.double) / p),5)


        for i in range(r, image.shape[0] - r):
            for j in range(r, image.shape[1] - r):
                resultCode = None
                if typeLBP == 'original':
                    if template.underFlow is None or template.overFlow is None:
                        resultCode = self.generateCode(image[i - 1:i + 2, j - 1:j + 2], np.array([1, 1]),typeMeasurement,firstLayer=firstLayer)
                    else:
                        resultCode = self.generateCode(image[i - 1:i + 2, j - 1:j + 2], np.array([1, 1]), typeMeasurement,template.overFlow[i - 1:i + 2, j - 1:j + 2],template.underFlow[i - 1:i + 2, j - 1:j + 2],firstLayer=firstLayer)
                elif typeLBP == 'pr':
                    if template.underFlow is None or template.overFlow is None:
                        resultCode = self.generateCodePR(image[i - r:i + (r + 1), j - r:j + (r + 1)], np.array([r, r]),xPositions, yPositions, typeMeasurement,firstLayer=firstLayer)
                    else:
                        resultCode = self.generateCodePR(image[i - r:i + (r + 1), j - r:j + (r + 1)], np.array([r, r]), xPositions,yPositions, typeMeasurement,template.overFlow[i - r:i + (r + 1), j - r:j + (r + 1)],template.underFlow[i - r:i + (r + 1), j - r:j + (r + 1)],firstLayer=firstLayer)

                if not template is None:
                    template.layersChar[i][j] = resultCode

                returnValue[0].append(resultCode[0])
                returnValue[1].append(resultCode[1])
                returnValue[2].append(resultCode[2])
                returnValue[3].append(resultCode[3])

        return returnValue

    def cropImage(self, image, le, re, no):
        distance = list(map(operator.sub, le[:2], re[:2]))
        distance = int(math.sqrt((distance[0] ** 2) + (distance[1] ** 2)))
        points_crop = (no[0] - distance, no[1] - distance, no[0] + distance, no[1] + distance)
        points_crop = list(map(int, points_crop))
        return image.crop((points_crop[0], points_crop[1], points_crop[2], points_crop[3]))

    def setupTemplate(self, template):
        '''
        if (type(template) is not LFWTemplate) and (type(template) is not FRGCTemplate) and (type(template) is not BosphorusTemplate):
            template.loadMarks('3DObj')
            subject = "%04d" % (template.itemClass)
            template3Dobj = template.rawRepr.split(os.path.sep)[:-3]
            folderType = template3Dobj[template3Dobj.index(subject) + 1]
            a, b, c, y = loadOBJ(os.path.join(os.path.sep.join(template3Dobj), '3DObj','depth_' + subject + '_' + folderType + '_' + template.typeTemplate + '.obj'))
            template.image = c
        '''
        # template.image = self.cropImage(template.image,template.faceMarks[0],template.faceMarks[1],template.faceMarks[2]).convert('L')
        # template.image = np.asarray(template.image.resize((64,64), im.ANTIALIAS))
        return template

    def cleanupTemplate(self, template):
        if (type(template) is not FRGCTemplate) and (type(template) is not BosphorusTemplate) and (len(template.image) > 0):
            template.layersChar = np.zeros((len(template.image), len(template.image[0]), 4))
            template.image = im.fromarray(np.array(template.image, dtype=np.uint8))
            template.image = template.image.rotate(-180)
            template.save(True)
        return template

    def featureExtraction(self, points=None, radius=None, paralelCalling=False,layersUtilize = [1,2,3,4],forceImage=True,typeMeasurement='Normal',procs=10,masks=False,firstLayer='lbp'):
        self.quantityItensProcessing = sum([len(d.templates) for d in self.databases])
        self.feitos = 0
        if paralelCalling:
            poolCalling = Pool(processes=procs)
            for database in self.databases:
                dataForParCal = [{'template': t, 'points': points, 'radius': radius, 'layersUtilize' : layersUtilize,'forceImage' : forceImage,'typeMeasurement' : typeMeasurement,'masks' : masks,'firstLayer' : firstLayer} for t in database.templates]
                responses = poolCalling.map(unwrap_self_f_feature, zip([self] * len(dataForParCal), dataForParCal))
                for i in range(len(responses)):
                    database.templates[i].features = responses[i][1]
        else:
            for database in self.databases:
                for template in database.templates:
                    dataForParCal = {'points': points, 'radius': radius, 'template': template, 'layersUtilize' : layersUtilize,'forceImage' : forceImage,'typeMeasurement' : typeMeasurement,'masks' : masks,'firstLayer' : firstLayer}
                    a, template.features = self.doFeatureExtraction(dataForParCal)

    def progessNumber(self):
        self.feitos += 1
        print("%d completed from %d" % (self.feitos,self.quantityItensProcessing))

    def localcall(self,parameters):
        template = parameters['template']
        if parameters['forceImage'] or not template.isFileExists(self.fullPathGallFile):
            points = parameters['points']
            radius = parameters['radius']
            imgCroped = np.asarray(template.image).astype(np.int64)
            uniArray = None
            if template.layersChar is None:
                template.layersChar = np.full((imgCroped.shape[0], imgCroped.shape[1], 4),255)

            if (not points is None) and (not radius is None):
                uniArray = generateArrayUniform(points)
                desc = self.generateImageDescriptor(imgCroped, p=points, r=radius,typeLBP='pr',template=template,typeMeasurement=parameters['typeMeasurement'],masks=parameters['masks'],firstLayer=parameters['firstLayer'])
                template.saveMasks('overflowMasks_pr','overflow')
                template.saveMasks('underflowMasks_pr', 'underflow')
            else:
                desc = self.generateImageDescriptor(imgCroped,template=template,typeMeasurement=parameters['typeMeasurement'],masks=parameters['masks'],firstLayer=parameters['firstLayer'])
                template.saveMasks('overflowMasks','overflow')
                template.saveMasks('underflowMasks', 'underflow')

            if self.generateImagesTrainig:
                saving = template.saveImageTraining(False, self.fullPathGallFile)
            offsetx = int(math.ceil(imgCroped.shape[0] / float(self.windowSize)))
            offsety = int(math.ceil(imgCroped.shape[1] / float(self.windowSize)))
            fullImageDescriptor = []
            imgCroped = template.layersChar[1:99,1:99,:]
            for i in range(0, imgCroped.shape[0], offsetx):
                for j in range(0, imgCroped.shape[1], offsety):
                    windowPiece = imgCroped[i:(i + offsetx), j:(j + offsety)]
                    reshapedWindow = [[],[],[],[]]
                    for wpx in range(windowPiece.shape[0]):
                        for wpy in range(windowPiece.shape[1]):
                            for wpz in range(windowPiece.shape[2]):
                                reshapedWindow[wpz].append(windowPiece[wpx,wpy,wpz])

                    for idxLayer in parameters['layersUtilize']:
                        if points is None:
                            fullImageDescriptor += generateHistogram(reshapedWindow[idxLayer-1], self.binsize)
                        else:
                            fullImageDescriptor += generateHistogramUniform(reshapedWindow[idxLayer - 1], points,uniArray)

            self.progessNumber()
            return saving, fullImageDescriptor
        else:
            self.progessNumber()
            return None, None