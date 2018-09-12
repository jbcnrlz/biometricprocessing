from LFW                     import *
from mlbp                    import *
from GeneratePCDLFW          import *
from RotateFaceLFW           import *
from GenerateNewDepthMapsLFW import *
from SymmetricFilling        import *
from SegmentFace             import *

if __name__ == '__main__':

    gallery = LFW('/home/joaocardia/Dropbox/pesquisas/unrestricted/feitas')    

    gallery.feedTemplates(True,'jpg',[str(i) for i in range(12)],'face')
    #gallery.loadRotatedFaces([10,20,30,-10,-20,-30])
    tdlbp = MahotasLBP(3,14,[gallery])

    paths = tdlbp.featureExtraction()
    
    galeryData = gallery.generateDatabaseFile(
        '/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_lbp_lfw_12_rotate_traditional.txt'
    )
    '''
    pathFaces = '/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_lbp_lfw_12_rotate_traditional.txt'
    f = open(pathFaces,'w')
    for t in paths:
        f.write(t + '\n')
    f.close()
    '''
    