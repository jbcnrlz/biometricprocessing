import operator, math, numpy as np, os, random, re, itertools, matplotlib.pyplot as plt, smtplib
from scipy.spatial.distance import euclidean
from PIL import Image as im
from textwrap import wrap
from sklearn.metrics import confusion_matrix
from email.message import EmailMessage
from yaml import load

def sendEmailMessage(subject,message):
    config = None
    with open("emailConfig.yaml", 'r') as stream:
        config = load(stream)

    # Create the container email message.
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = config['from']
    msg['To'] = config['to']
    msg.set_content(message)

    server = smtplib.SMTP(config['server'],int(config['port']))
    server.login(config['login'],config['password'])
    server.send_message(msg)
    server.quit()

def zFunc(t,A):
    return 1 / (1 + np.exp(-A * np.log(2 + np.sqrt(3)) * t ))

def wFunc(t,A):
    return (1 / (zFunc(1/A,A) - 0.5)) * (zFunc(t,A) - 0.5)

def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.Figure(figsize=(4, 4), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    imcf = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=3, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    return fig


def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0
    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def generateHistogram(data,nbins,labels=False):
    histogram = [0] * nbins
    if (len(data) <= 0):
        return histogram
    indexMaxValue, maxValue = max(enumerate(data), key=operator.itemgetter(1))
    step = maxValue / nbins
    start = 0
    stop = start + step
    labelOrder = []
    for bin in range(nbins):
        for n in data:
            if (n >= start) and (n < stop):
                histogram[bin] += 1
                labelOrder.append(bin)
            elif (n >= start) and (n <= stop) and ((bin + 1) == nbins):
                histogram[bin] += 1
                labelOrder.append(bin)
        start = stop
        if (bin + 2) == nbins:            
            stop = 1.0
        else:
            stop = start + step
    if labels:
        return labelOrder
    else:
        return minmax(histogram)

def scaleValues(a,b,data):
    newData = np.zeros(data.shape)
    diffValue = b-a
    minValue = np.amin(data)
    maxValue = np.amax(data)
    divValue = maxValue - minValue
    for i in range(newData.shape[0]):
        for j in range(newData.shape[1]):
            newData[i][j] = ((diffValue*(data[i][j] - minValue)) / divValue) + a

    return newData

def minmax(itens):
    real_list = list(itens)
    itens.sort()
    minval = float(itens[0])
    maxval = float(itens[len(itens) - 1])
    returnOrdered = []
    divisor = maxval - minval if (maxval - minval) > 0 else 1
    for i in real_list:
        returnOrdered.append((i - minval) / divisor)
    return returnOrdered

def loadOBJ(filename,outputFacet=False):
    numVerts = 0
    verts = []
    norms = []
    vertsOut = []
    normsOut = []
    idxsVerts = []
    for line in open(filename, "r"):
        vals = line.split()
        if (len(vals) > 0):
            if vals[0] == "v":
                v = list(map(float, vals[1:4]))
                verts.append(v)
            if vals[0] == "vn":
                n = list(map(float, vals[1:4]))
                norms.append(n)
            if vals[0] == "f":
                idxsVerts.append([])
                for f in vals[1:]:
                    w = f.split("/")
                    # OBJ Files are 1-indexed so we must subtract 1 below
                    vertsOut.append(list(verts[int(w[0])-1]))
                    idxsVerts[-1].append(int(w[0]) - 1)
                    normsOut.append(list(norms[int(w[2])-1]))
                    numVerts += 1
    if outputFacet:
        return vertsOut, normsOut, verts, norms, idxsVerts
    else:
        return vertsOut, normsOut, verts, norms

def isUniform(number):
    changes = 0
    last = number[0]
    for n in number:
        if (n != last):
            last = n
            changes += 1

        if changes > 2:
            return False
    return True

def generateArrayUniform(points):
    maxNumberQntde = int(math.pow(2,points))
    currBinNumer = ['0'] * (points)
    uniformNumbersDecimal = {}
    for i in range(maxNumberQntde):
        changesInNumber = 0
        for j in range(len(currBinNumer) - 1):
            if (currBinNumer[j] != currBinNumer[j+1]):
                changesInNumber += 1

            if(changesInNumber > 2):
                break

        if (changesInNumber <= 2):
            uniformNumbersDecimal[int(''.join(currBinNumer),2)] = 0

        currBinNumer = [ k for k in format(int(''.join(currBinNumer),2)+0b1,'#0'+str(points+2)+'b')[2:]]
    uniformNumbersDecimal['n'] = 0
    return uniformNumbersDecimal

def generateHistogramUniform(data,numberPoints,histogram=None):
    if histogram is None:
        histogram = generateArrayUniform(numberPoints)
    else:
        histogram = dict(histogram)

    for d in data:
        if d in histogram.keys():
            histogram[d] += 1
        else:
            histogram['n'] += 1

    lastArgument = histogram['n']
    del histogram['n']
    returnOrderedHistogram = [histogram[key] for key in sorted(histogram.keys())]+[lastArgument]
    return minmax(returnOrderedHistogram)

def outputObj(points,fileName):
    f = open(fileName,'w')
    f.write('g \n')
    for p in points:
        if (len(p) == 2):
            f.write('v ' + ' '.join(map(str,p[0])) + '\n')
        else:
            f.write('v ' + ' '.join(map(str,p)) + '\n')
    f.write('g 1 \n')
    f.close()

def getArbitraryMatrix(axis,theta):
    origin = np.array([0,0,0])
    if (axis != origin).any():
        axis = axis/math.sqrt(np.dot(axis,axis))
    a = math.cos(theta/2)
    b,c,d = -axis*math.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def getRotationMatrix(theta):
    return np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])

def getRotationMatrix3D(angle,matrix):
    cosResult = np.cos(angle)
    sinResult = np.sin(angle)
    if matrix == 'x':
        return np.array([
                    (1 ,0, 0, 0),
                    (0 , cosResult, -sinResult, 0),                
                    (0 , sinResult,  cosResult, 0),
                    (0 ,         0,          0, 1)
                ])
    elif matrix == 'y':
        return np.array([
                    (cosResult , 0, sinResult, 0),
                    (0         , 1, 0        , 0),
                    (-sinResult, 0, cosResult, 0),
                    (0         , 0,         0, 1)                
                ])
    else:
        return np.array([
                    (cosResult ,-sinResult, 0, 0),
                    (sinResult , cosResult, 0, 0),                
                    (0         ,         0, 1, 0),
                    (0         ,         0, 0, 1)
                ])

def getRegionFromCenterPoint(center,radius,points):
    centerPoint = points[center]
    sortedList = list(points)
    neighbors = []
    for x in range(center-1,-1,-1):
        if ((centerPoint != sortedList[x]).all() and (sortedList[x] != [0.0,0.0,0.0]).all()):
            distancePoints = euclidean(np.array(centerPoint[0:2]),np.array(sortedList[x][0:2]))
            if distancePoints <= radius:
                neighbors.append(sortedList[x])

    for x in range(center + 1,len(sortedList)):
        if ((centerPoint != sortedList[x]).all() and (sortedList[x] != [0.0,0.0,0.0]).all()):
            distancePoints = euclidean(np.array(centerPoint[0:2]),np.array(sortedList[x][0:2]))
            if distancePoints <= radius:
                neighbors.append(sortedList[x])
    return neighbors

def findPointIndex(points,pointIndex):
    smallerDistance = [100000000000000000000000000000000000000000000000000000000,0]
    for p in range(len(points)):
        accDist = 0
        for i in range(len(pointIndex)):
            accDist += euclidean(points[p][i],pointIndex[i])
            '''
            if (points[p][0] == pointIndex[0] and points[p][1] == pointIndex[1] and points[p][2] == pointIndex[2]):
                return p
            elif(euclidean(np.array(points[p]),np.array(pointIndex)) < smallerDistance[0]):
                smallerDistance[0] = euclidean(np.array(points[p]),np.array(pointIndex))
                smallerDistance[1] = p
            '''
        if accDist < smallerDistance[0]:
            smallerDistance[0] = accDist
            smallerDistance[1] = p
    
    return smallerDistance[1]

def mergeArraysDiff(arrA,arrB):
    for k in range(arrA.shape[2]):
        idxOuter = 0
        for i in range(arrA.shape[0]):
            for j in range(arrA.shape[1]):
                if (idxOuter < len(arrB[k])):
                    arrA[i,j,k] = arrB[k][idxOuter]

                idxOuter += 1

    return arrA

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def getDirectoriesInPath(path):
    return [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]

def getFilesInPath(path,onlyFiles=True):
    if onlyFiles:
        return [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]

def generateData(pathFiles,extension='png'):
    returnDataImages = []
    returnDataClass = []
    filesOnPath = getFilesInPath(pathFiles)
    for f in filesOnPath:
        if f[-3:] == extension:
            returnDataImages.append(f)
            classNumber = f.split(os.path.sep)[-1]
            if extension in ['png','npy']:
                classNumber = classNumber.split('_')[0]
            elif classNumber[0] == 'b':
                classNumber = classNumber[2:5]
            else:
                classNumber = classNumber.split('_')[1]
            returnDataClass.append(int(classNumber))


    return returnDataImages, returnDataClass

def generateExperimentDataPattern(imageData,classesData,patternProbe,patternGallery):
    foldProbe = []
    foldProbeClasses = []
    foldGallery = []
    foldGalleryClasses = []
    for i in range(len(imageData)):
        isProbe = False
        fileName = imageData[i].split(os.path.sep)[-1]
        for p in patternProbe:
            if re.match(p, fileName):
                foldProbe.append(imageData[i])
                foldProbeClasses.append(classesData[i])
                isProbe = True
                break

        if not isProbe:
            if patternGallery is None:
                foldGallery.append(imageData[i])
                foldGalleryClasses.append(classesData[i])
            else:
                for pg in patternGallery:
                    if re.match(pg,fileName):
                        foldGallery.append(imageData[i])
                        foldGalleryClasses.append(classesData[i])

    return foldGallery, foldGalleryClasses, foldProbe, foldProbeClasses


def generateFoldsOfData(fq,imageData,classesData):
    foldSize = int(len(imageData) / fq)
    foldResult = []
    alreadyWentFold = []
    for foldNumber in range(fq):
        print('Fazendo fold ' + str(foldNumber+1))
        foldChoices = random.sample([i for i in range(len(imageData)) if i not in alreadyWentFold], foldSize)
        alreadyWentFold = alreadyWentFold + foldChoices
        foldProbe = []
        foldProbeClasses = []
        foldGallery = []
        foldGalleryClasses = []
        for i in range(len(imageData)):
            if i in foldChoices:
                foldProbe.append(imageData[i])
                foldProbeClasses.append(classesData[i])
            else:
                foldGallery.append(imageData[i])
                foldGalleryClasses.append(classesData[i])

        foldResult.append([foldGallery, foldGalleryClasses, foldProbe, foldProbeClasses])

    return foldResult

def generateImageData(paths,resize=None,silent=False):
    returningPaths = []
    for p in paths:
        if not silent:
            print('Loading image '+p)
        ni = im.open(p)
        if not resize is None:
            ni = ni.resize(resize,im.ANTIALIAS)

        ni = np.array(ni)
        returningPaths.append(np.array(ni))
    return np.array(returningPaths)

def loadFoldFromFolders(pathFolders):
    inFolder = None
    try:
        inFolder = list(map(int,getDirectoriesInPath(pathFolders)))
        inFolder.sort()
        inFolder = list(map(str, inFolder))
    except:
        inFolder = getDirectoriesInPath(pathFolders)
    returnFolders = []
    for inF in inFolder:
        returnFolders.append([[],[],[],[]])
        types = getDirectoriesInPath(os.path.join(pathFolders,inF))
        for t in types:
            filesForFold = getFilesInPath(os.path.join(pathFolders,inF,t))
            for ffolder in filesForFold:
                currFile = ffolder.split(os.path.sep)[-1]

                cClass = int(currFile.split('_')[0])
                if t == 'probe':
                    returnFolders[-1][2].append(ffolder)
                    returnFolders[-1][3].append(cClass)
                else:
                    returnFolders[-1][0].append(ffolder)
                    returnFolders[-1][1].append(cClass)

    return returnFolders

if __name__ == '__main__':
    print(generateArrayUniform())

