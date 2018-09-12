import operator, math, numpy as np, os
from scipy.spatial.distance import euclidean

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

def loadOBJ(filename):
    numVerts = 0
    verts = []
    norms = []
    vertsOut = []
    normsOut = []
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
                for f in vals[1:]:
                    w = f.split("/")
                    # OBJ Files are 1-indexed so we must subtract 1 below
                    vertsOut.append(list(verts[int(w[0])-1]))
                    normsOut.append(list(norms[int(w[2])-1]))
                    numVerts += 1
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

def generateHistogramUniform(data,numberPoints):
    histogram = generateArrayUniform(numberPoints)
    for d in data:
        if d in histogram.keys():
            histogram[d] += 1
        else:
            histogram['n'] += 1
    return list(histogram.values())

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

def getFilesInPath(path):
    return [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

if __name__ == '__main__':
    print(generateArrayUniform())

