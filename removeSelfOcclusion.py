import numpy as np
from scipy.spatial import ConvexHull

def getRotationMatrix(angle,matrix):
    cosResult = np.cos(np.radians(angle))
    sinResult = np.sin(np.radians(angle))
    if matrix == 'x':
        return np.array([
            (1 ,0, 0),
            (0 , cosResult, -sinResult),
            (0 , sinResult,  cosResult)
        ])
    elif matrix == 'y':
        return np.array([
            (cosResult , 0, sinResult),
            (0         , 1, 0        ),
            (-sinResult, 0, cosResult),
        ])
    else:
        return np.array([
            (cosResult ,-sinResult, 0),
            (sinResult , cosResult, 0),
            (0         ,         0, 1),
        ])

def rotatePointCloud(P, R, t):
    return np.dot(R, np.transpose(P))

def estimateVis_vertex(vertex, R, C_dist, r):
    viewPoint_front = np.array([0,0,C_dist]).reshape(1,3, order='F')
    viewPoint = np.transpose(rotatePointCloud(viewPoint_front, R, []))
    visIdx = HPR(vertex, viewPoint, r)
    return visIdx # controllare la funzione HPR

def HPR(p, C, param):
    dim = p.shape[1]
    numPts = p.shape[0]
    p = p - np.tile(C,(numPts,1))
    #normP = np.sqrt(p.dot(p))
    normP = np.linalg.norm(p, axis=1)
    normP = normP.reshape(normP.shape[0], 1, order='F')
    app = np.amax(normP)*(np.power(10,param))
    R = np.tile(app,(numPts,1))

    P = p + 2*np.tile((R-normP),(1,dim))*p/np.tile(normP,(1,dim))
    _zeros = np.zeros((1,dim))
    vect_conv_hull = np.vstack([P,_zeros])
    hull = ConvexHull(vect_conv_hull)
    visiblePtInds = np.unique(hull.vertices)
    for i in range(visiblePtInds.shape[0]):
        visiblePtInds[i] =  visiblePtInds[i] - 1
        if visiblePtInds[i] == (numPts + 1):
            visiblePtInds.remove(i)

    return visiblePtInds.reshape(visiblePtInds.shape[0], 1, order='F')

def loadOBJ(filename, outputFacet=False):
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
                    vertsOut.append(list(verts[int(w[0]) - 1]))
                    idxsVerts[-1].append(int(w[0]) - 1)
                    normsOut.append(list(norms[int(w[2]) - 1]))
                    numVerts += 1
    if outputFacet:
        return vertsOut, normsOut, verts, norms, idxsVerts
    else:
        return vertsOut, normsOut, verts, norms

def outputObj(points, fileName):
    f = open(fileName, 'w')
    f.write('g \n')
    for p in points:
        if (len(p) == 2):
            f.write('v ' + ' '.join(map(str, p[0])) + '\n')
        else:
            f.write('v ' + ' '.join(map(str, p)) + '\n')
    f.write('g 1 \n')
    f.close()

def main():
    a, b, c, d = loadOBJ('depth_0001_s1_LightOn_rotate_-30.obj')
    rmtx = getRotationMatrix(-30,'x')
    c = np.array(c)
    visparts = estimateVis_vertex(c,rmtx,300,4)
    visparts = visparts[visparts >=0]
    outputObj(c[visparts],'teste.obj')

if __name__ == '__main__':
    main()