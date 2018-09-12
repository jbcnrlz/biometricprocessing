from LFW                     import *
from tdlbp                   import *
from acdn                    import *
from GeneratePCDLFW          import *
from RotateFaceLFW           import *
from GenerateNewDepthMapsLFW import *
from SymmetricFilling        import *
from SegmentFace             import *
import scipy.ndimage as ndimage, logging, sys

if __name__ == '__main__':

    gallery = LFW('/home/joaocardia/Dropbox/pesquisas/LFW/feitas')    
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

    gallery.feedTemplates(True,'bmp',[str(i) for i in range(2)],'face')
    gallery.loadRotatedFaces([10,20,30,-10,-20,-30],['y','x'])
    tdlbp = ThreeDLBP(8,14,[gallery])
    #tdlbp.preProcessingSteps = GeneratePCDLFW()
    #tdlbp.preProcessingSteps = SegmentFace()
    #tdlbp.preProcessingSteps = SymmetricFilling()
    tdlbp.preProcessingSteps = GenerateNewDepthMapsLFW()
    #tdlbp.preProcessingSteps = RotateFaceLFW(['y','x'],False)
    #tdlbp.preProcessing(True,False)
    
    
    paths = tdlbp.featureExtraction()
    
    galeryData = gallery.generateDatabaseFile(
        '/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/buceltilde_quali.txt'
    )

    pathFaces = '/home/joaocardia/Dropbox/pesquisas/classificador/SVMTorch_linux/test_data/gallery_3dlbp_lfw_faces.txt'
    f = open(pathFaces,'w')
    for t in paths:
        f.write(t + '\n')
    f.close()
    