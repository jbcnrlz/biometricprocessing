from LFW                     import *
from acdn                    import *
from GeneratePCDLFW          import *
from RotateFaceLFW           import *
from GenerateNewDepthMapsLFW import *
from SymmetricFilling        import *
from SegmentFace             import *
import scipy.ndimage as ndimage, logging, sys

if __name__ == '__main__':

    gallery = LFW('/home/joaocardia/Dropbox/pesquisas/unrestricted/feitas')    
    '''
    gallery.feedTemplates(True,'pcd',[str(i) for i in range(12)],'face_segmented')
    ac = acdn([gallery])
    #ac.preProcessingSteps = GeneratePCDLFW()
    #ac.preProcessing(True,False)
    ac.featureExtraction()

    galeryData = gallery.generateDatabaseFile(
        '/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_acdnp_lfw_12.txt'
    )
    '''
    gallery.feedTemplates(True,'pcd',[str(i) for i in range(12)],'face')
    #gallery.loadRotatedFaces([10,20,30,-10,-20,-30])
    ac = acdn([gallery])
    ac.featureExtraction()

    galeryData = gallery.generateDatabaseFile(
        '/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_acdnp_lfw_12_roate_traditional.txt'
    )
    