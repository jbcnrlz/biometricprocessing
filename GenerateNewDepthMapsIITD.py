from baseClasses.PreProcessingStep import *
import math, numpy as np, os
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import norm
from helper.functions import scaleValues

class GenerateNewDepthMapsIITD(PreProcessingStep):

    def generateImage(self,image):
        x = image[:, 0] -1
        y = image[:, 1]-1
        z = image[:, 2]

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        xnodes = np.linspace(xmin,xmax,99)
        xnodes[-1] = xmax
        ynodes = np.linspace(ymin,ymax,99)
        ynodes[-1] = ymax

        dx = np.diff(xnodes)
        dy = np.diff(ynodes)
        nx = len(xnodes)
        ny = len(ynodes)
        ngrid = nx*ny

        xscale = np.mean(dx)
        yscale = np.mean(dy)
        maskflag = 0
        maxiter = min([10000,nx*ny])

        if xmin < xnodes[0]:
            xnodes[0] = xmin
        if xmax > xnodes[-1]:
            xnodes[-1] = xmax
        if ymin < ynodes[0]:
            ynodes[0] = ymin
        if ymax > ynodes[-1]:
            ynodes[-1] = ymax

        indx = np.digitize(x,xnodes)-1
        indy = np.digitize(y,ynodes)-1
        k = indx == nx
        indx[k] = indx[k] - 1
        k = indy == ny
        indy[k] = indy[k] - 1
        ind = indy + ny*(indx)

        tx = (x - xnodes[indx]) / dx[indx]
        tx[tx < 0] = 0
        tx[tx > 1] = 1
        ty = (y - ynodes[indy]) / dy[indy]
        ty[ty < 0] = 0
        ty[ty > 1] = 1

        k = (tx > ty)
        n = len(x)
        L = np.ones((n,1)).flatten()
        L[k] = ny

        t1 = np.minimum(tx,ty)
        t2 = np.maximum(tx,ty)

        repmatDivide = np.tile(np.arange(0,n),(3))
        block1 = np.block([ind,ind+ny+1,ind+L])
        block2 = np.block([1-t2,t1,t2-t1])

        A = csr_matrix((block2, (repmatDivide,block1)), shape=(n,ngrid))
        rhs = z

        smoothparam = 1
        xyRelativeStiffness = np.ones((2,1))

        i,j = np.mgrid[1:nx+1,2:ny]
        i = i.T
        j = j.T
        ind = j.flatten() + ny *(i.flatten()-1)
        dy1 = dy[j.flatten()-2]/yscale
        dy2 = dy[j.flatten()-1]/yscale

        repmatDivide = np.tile(ind,(3))
        block1 = np.block([ind-1,ind,ind+1])
        block2 = xyRelativeStiffness[1] * np.block([-2/(dy1 * (dy1+dy2)), 2/(dy1 * dy2), -2 / (dy2 * (dy1 + dy2)) ])
        Areg = csr_matrix((block2, (repmatDivide, block1)), shape=(ngrid+1, ngrid+1))

        i,j = np.mgrid[2:nx-1,1:ny]
        i = i.T
        j = j.T
        ind = j.flatten() + ny *(i.flatten()-1)
        dx1 = dx[i.flatten()-2]/xscale
        dx2 = dx[i.flatten()-1]/xscale
        repmatDivide = np.tile(ind, (3))
        block1 = np.block([ind-ny,ind,ind+ny])
        block2 = xyRelativeStiffness[0] * np.block([-2/(dx1 * (dx1+dx2)), 2/(dx1 * dx2), -2 / (dx2 * (dx1 + dx2)) ])
        Areg2 = csr_matrix((block2, (repmatDivide, block1)), shape=(ngrid + 1, ngrid + 1))
        Areg = hstack([Areg,Areg2])
        nreg = Areg.shape[1]

        NA = norm(A)
        NR = norm(Areg,axis=1)
        Areg = Areg * (smoothparam*NA/NR)
        A = hstack([A,Areg])

        print('oi')

    def doPreProcessing(self,template):
        template.image = np.array(self.generateImage(template.image))
        template.image = scaleValues(0, 255, template.image)
        template.save()
        return template


if __name__ == '__main__':
    from helper.functions import loadOBJ

    a,b,c,d = loadOBJ('depth_0001_s1_LightOn_symmetricfilled.obj')
    gnd = GenerateNewDepthMapsIITD()
    a = gnd.generateImage(np.array(c))