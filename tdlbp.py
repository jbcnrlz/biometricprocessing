import logging, math, operator
from scipy.special import expit
from FRGCTemplate import *
from LFWTemplate import *
from baseClasses.BiometricProcessing import *
from helper.functions import generateHistogram, bilinear_interpolation, mergeArraysDiff, printProgressBar


class ThreeDLBP(BiometricProcessing):

    def __init__(self,windowsize,binsize,database):
        self.windowSize = windowsize
        self.binsize = binsize
        self.databases = database
        self.methodName = '3DLBP'
        super().__init__()

    def saveDebug(self,folder,data):
        with open(folder+'/subtraction.txt', 'a') as rty:
            rty.write(str(data) + '\n')

    def generateCode(self,image,center,typeOp='Normal'):
        idxs = [
            (center[0]-1,center[1]-1),
            (center[0]-1,center[1]),
            (center[0]-1,center[1]+1),
            (center[0],center[1]-1),
            (center[0],center[1]+1),
            (center[0]+1,center[1]-1),
            (center[0]+1,center[1]),
            (center[0]+1,center[1]+1)
        ]
        layers = [[],[],[],[]]
        for i in idxs:
            subraction = 0
            if typeOp == 'Normal':
                subraction = int(round(image[i[0]][i[1]] - image[center[0]][center[1]]))
                if subraction < -7:
                    subraction = -7
                elif subraction > 7:
                    subraction = 7
            else:
                subraction = image[i[0]][i[1]] - image[center[0]][center[1]]
                subraction = np.histogram(expit(subraction),bins=7,range=[0,1])[0]
                subraction = np.argwhere(subraction==1)[0][0]
            bin = '{0:03b}'.format(abs(subraction))
            layers[0].append(str(int(subraction >= 0)))
            layers[1].append(bin[0])
            layers[2].append(bin[1])
            layers[3].append(bin[2])
        for l in range(len(layers)):
            layers[l] = int(''.join(layers[l]),2)       
        return layers

    '''
    Gera indices quando usando P e R no 3DLBP ao inves da vizinhanca fixa de 3x3
    '''
    def generateIndexes(self,R,P,p):
        x = round(-R * np.sin( (2*np.pi*p) / P) + 1,5)
        y = round(R * np.cos( (2*np.pi*p) / P) + 1,5)
        return (x,y)

    def generateCodePR(self,image,center,P,R,type='Normal'):
        idxs = []
        for p in range(P):
            idxs.append(self.generateIndexes(R,P,p))

        layers = [[],[],[],[]]
        for i in idxs:
            subraction = 0
            if (not i[0].is_integer() or not i[1].is_integer()):
                xidxs = [math.floor(i[0]),math.ceil(i[0])]
                yidxs = [math.floor(i[1]),math.ceil(i[1])]
                imageDataBil = [
                    (xidxs[0],yidxs[0],image[xidxs[0]][yidxs[0]]),
                    (xidxs[0],yidxs[1],image[xidxs[0]][yidxs[1]]),
                    (xidxs[1],yidxs[0],image[xidxs[1]][yidxs[0]]),
                    (xidxs[1],yidxs[1],image[xidxs[1]][yidxs[1]]),
                ]
                #subraction = int(round(bilinear_interpolation(i[0],i[1],imageDataBil)) - image[center[0]][center[1]])
                subraction = bilinear_interpolation(i[0],i[1],imageDataBil) - image[center[0]][center[1]]
            else:
                subraction = image[int(i[0])][int(i[1])] - image[center[0]][center[1]]

            if type == 'Normal':
                subraction = int(round(subraction))
                
                #self.saveDebug('debug_subs',subraction)

                if subraction < -7:
                    subraction = -7
                elif subraction > 7:
                    subraction = 7
            else:
                subraction = np.histogram(expit(subraction),bins=7,range=[0,1] )[0]                
                subraction = np.argwhere(subraction==1)[0][0]

            bin = '{0:03b}'.format(abs(subraction))         
            layers[0].append(str(int(subraction >= 0)))
            layers[1].append(bin[0])
            layers[2].append(bin[1])
            layers[3].append(bin[2])
        for l in range(len(layers)):
            layers[l] = int(''.join(layers[l]),2)       
        return layers

    def generateImageDescriptor(self,image,p=8,r=1,typeLBP='original',typeMeasurement='Normal'):
        returnValue = [[],[],[],[]]
        #returnValue = self.generateCodePR(image,[1,1],p,r)

        log= logging.getLogger( "ThreeDLBP.generateImageDescriptor" )
        #log.debug("Valor de retorno"+str(returnValue))

        
        for i in range(r,image.shape[0]-r):
            for j in range(r,image.shape[1]-r):
        #for i in range(1,image.shape[0]-1):
        #    for j in range(1,image.shape[1]-1):
                #resultCode = self.generateCodePR(image[i-r:i+(1+r),j-r:j+(1+r)],np.array([r,r]),p,r,'Sigmoide')
                #resultCode = self.generateCodePR(np.array([[1,2,3],[4,5,6],[7,8,9]]),np.array([r,r]),p,r)
                #resultCode = self.generateCode(image[i-r:i+(1+r),j-r:j+(1+r)],np.array([1,1]))
                if typeLBP == 'original':
                    resultCode = self.generateCode(image[i-1:i+2,j-1:j+2],np.array([1,1]),typeMeasurement)
                elif typeLBP == 'pr':
                    resultCode = self.generateCodePR(image[i-r:i+(r+1),j-r:j+(r+1)],np.array([r,r]),p,r,typeMeasurement)
                returnValue[0].append(resultCode[0])
                returnValue[1].append(resultCode[1])
                returnValue[2].append(resultCode[2])
                returnValue[3].append(resultCode[3])
        
        return returnValue

    def cropImage(self,image,le,re,no):
        distance = list(map(operator.sub,le[:2],re[:2]))
        distance = int(math.sqrt((distance[0]**2) + (distance[1]**2)))
        points_crop = (no[0] - distance,no[1] - distance,no[0] + distance,no[1] + distance)
        points_crop = list(map(int,points_crop))   
        return image.crop((points_crop[0],points_crop[1],points_crop[2],points_crop[3]))

    def setupTemplate(self,template):
        if (not type(template) is LFWTemplate) and (not type(template) is FRGCTemplate):
            template.loadMarks('3DObj')
            subject = "%04d" % (template.itemClass)
            template3Dobj = template.rawRepr.split(os.path.sep)[:-3]
            folderType = template3Dobj[template3Dobj.index(subject) + 1]
            a, b, c, y = loadOBJ(os.path.join(os.path.sep.join(template3Dobj),'3DObj','depth_'+subject+'_'+folderType+'_'+template.typeTemplate+'.obj'))
            template.image = c

        #template.image = self.cropImage(template.image,template.faceMarks[0],template.faceMarks[1],template.faceMarks[2]).convert('L')
        #template.image = np.asarray(template.image.resize((64,64), im.ANTIALIAS))        
        return template

    def cleanupTemplate(self,template):
        if (not type(template) is FRGCTemplate):
            template.layersChar = np.zeros((len(template.image),len(template.image[0]),4))        
            template.image = im.fromarray(np.array(template.image,dtype=np.uint8))
            template.image = template.image.rotate(-180)
            template.save(True)
        return template

    def featureExtraction(self,points,radius,paralelCalling=False):
        if paralelCalling:
            poolCalling = Pool()
            for database in self.databases:
                dataForParCal = [{'template':t,'points': points,'radius': radius} for t in database.templates]
                responses = poolCalling.map(unwrap_self_f_feature,zip([self]*len(dataForParCal), dataForParCal))
                for i in range(len(responses)):
                    database.templates[i].features = responses[i][1]
        else:
            for database in self.databases:
                for template in database.templates:
                    dataForParCal = {'points': points, 'radius': radius, 'template': template}
                    self.doFeatureExtraction(dataForParCal)


    def localcall(self,parameters):
        print("Iniciando feature extraction")
        template = parameters['template']
        points = parameters['points']
        radius = parameters['radius']
        imgCroped = np.asarray(template.image).astype(np.int64)

        if template.layersChar is None:
            template.layersChar = np.zeros((imgCroped.shape[0], imgCroped.shape[1], 4))

        offsetx = int(math.ceil(imgCroped.shape[0] / float(self.windowSize)))
        offsety = int(math.ceil(imgCroped.shape[1] / float(self.windowSize)))
        fullImageDescriptor = []
        for i in range(0, imgCroped.shape[0], offsetx):
            for j in range(0, imgCroped.shape[1], offsety):
                desc = None
                if (not points is None) and (not radius is None):
                    desc = self.generateImageDescriptor(imgCroped[i:(i + offsetx), j:(j + offsety)], p=points,r=radius, typeLBP='pr')
                else:
                    desc = self.generateImageDescriptor(imgCroped[i:(i + offsetx), j:(j + offsety)])
                template.layersChar[i + 1:(i + offsetx - 1), j + 1:(j + offsety - 1), :] = mergeArraysDiff(template.layersChar[i + 1:(i + offsetx - 1), j + 1:(j + offsety - 1), :], desc)
                fullImageDescriptor += generateHistogram(desc[0], self.binsize) + generateHistogram(desc[1],self.binsize) + generateHistogram(desc[2], self.binsize) + generateHistogram(desc[3], self.binsize)
        template.features = fullImageDescriptor
        saving = template.saveImageTraining(False,self.fullPathGallFile)
        return saving , fullImageDescriptor
