from FRGCTemplate import *
from LFWTemplate import *
from BosphorusTemplate import *
from baseClasses.BiometricProcessing import *
from scipy.interpolate import interp2d
from helper.functions import fitPlane

class AngularFeatureExtraction(BiometricProcessing):

    def __init__(self, database,generateImages=True):
        self.databases = database
        self.methodName = 'Angular'
        self.generateImagesTrainig = generateImages
        self.sigmoid_table = {}
        super().__init__()

    def generateCode(self, image, center):
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
        points = []
        for i in idxs:
            points.append([i[0], i[1], image[i[0]][i[1]]])

        return round(self.getAnglePlaneAxis(np.array(points + [[center[0],center[1],image[center[0]][center[1]]]])))

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
        finalNormal = finalNormal / np.linalg.norm(finalNormal)
        cosAngleZ = np.dot(finalNormal, np.array([0, 0, 1]))
        return cosAngleZ

    def generateCodePR(self, image, center, xPositions, yPositions):
        image = np.ascontiguousarray(image, dtype=np.double)
        xPositions = xPositions + center[0]
        yPositions = yPositions + center[1]
        idxs = np.array([x for x in zip(xPositions,yPositions)])
        idxs = idxs[-3:].tolist() + idxs[0:-3].tolist()
        points = []
        for i in idxs:
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
            else:
                points.append([i[0], i[1], image[int(i[0])][int(i[1])]])

        angle = self.getAnglePlaneAxis(np.array(points + [[center[0],center[1],image[center[0]][center[1]]]]))
        subraction = np.histogram(angle, bins=256, range=[-1, 1])[0]
        return np.argwhere(subraction == 1)[0][0]

    def generateImageDescriptor(self, image, p=8, r=1, typeLBP='original', template=None):
        if typeLBP == 'pr':
            xPositions = np.round(- r * np.sin(2 * np.pi * np.arange(p, dtype=np.double) / p),5)
            yPositions = np.round(r * np.cos(2 * np.pi * np.arange(p, dtype=np.double) / p),5)

        for i in range(r, image.shape[0] - r):
            for j in range(r, image.shape[1] - r):
                resultCode = None
                if typeLBP == 'original':
                    resultCode = self.generateCode(image[i - 1:i + 2, j - 1:j + 2], np.array([1, 1]))
                elif typeLBP == 'pr':
                    resultCode = self.generateCodePR(image[i - r:i + (r + 1), j - r:j + (r + 1)], np.array([r, r]), xPositions,yPositions)

                if not template is None:
                    template.layersChar[i][j] = resultCode

    def setupTemplate(self, template):
        return template

    def cleanupTemplate(self, template):
        if (type(template) is not FRGCTemplate) and (type(template) is not BosphorusTemplate) and (len(template.image) > 0):
            template.layersChar = np.zeros((len(template.image), len(template.image[0])))
            template.image = im.fromarray(np.array(template.image, dtype=np.uint8))
            template.image = template.image.rotate(-180)
            template.save(True)
        return template

    def featureExtraction(self, points=None, radius=None, paralelCalling=False,forceImage=True,procs=10):
        self.quantityItensProcessing = sum([len(d.templates) for d in self.databases])
        self.feitos = 0
        if paralelCalling:
            poolCalling = Pool(processes=procs)
            for database in self.databases:
                dataForParCal = [{'template': t, 'points': points, 'radius': radius, 'forceImage' : forceImage} for t in database.templates]
                responses = poolCalling.map(unwrap_self_f_feature, zip([self] * len(dataForParCal), dataForParCal))
        else:
            for database in self.databases:
                for template in database.templates:
                    dataForParCal = {'points': points, 'radius': radius, 'template': template, 'forceImage' : forceImage}
                    a, template.features = self.doFeatureExtraction(dataForParCal)

    def progessNumber(self):
        self.feitos += 1
        print("%d completed from %d" % (self.feitos,self.quantityItensProcessing))

    def localcall(self,parameters):
        template = parameters['template']
        if parameters['forceImage'] or not template.isFileExists(self.fullPathGallFile,'bmp'):
            print(template.rawRepr)
            points = parameters['points']
            radius = parameters['radius']
            imgCroped = np.asarray(template.image).astype(np.int64)
            if template.layersChar is None:
                template.layersChar = np.full((imgCroped.shape[0], imgCroped.shape[1]),255)

            if (not points is None) and (not radius is None):
                self.generateImageDescriptor(imgCroped, p=points, r=radius,typeLBP='pr',template=template)
            else:
                self.generateImageDescriptor(imgCroped,template=template)

            if self.generateImagesTrainig:
                template.saveImageTraining(False, self.fullPathGallFile)

            return None, None
        else:
            return None, None