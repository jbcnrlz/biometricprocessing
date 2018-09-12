from Eurecom import *
from lbp_test import *
import scipy.ndimage as ndimage

if __name__ == '__main__':

    gallery = EurecomKinect('/home/jbcnrlz/Documents/eurecom/EURECOM_Kinect_Face_Dataset','s1','RGB',['Neutral'])
    gallery.feedTemplates()

    probe = EurecomKinect('/home/jbcnrlz/Documents/eurecom/EURECOM_Kinect_Face_Dataset','s1','RGB',['Neutral'])
    probe.feedTemplates()

    tdlbp = LBPMahotas(3,[gallery,probe])
    tdlbp.preProcessing()
    tdlbp.featureExtraction()
    resultados = tdlbp.matcher()
    
    print(resultados)

    '''
    sub1s1 = gallery.getTemplateFromSubject(1).pop()
    sub1s2 = probe.getTemplateFromSubject(1).pop()

    sub1s1.image = im.fromarray(ndimage.gaussian_filter(np.asarray(sub1s1.image), sigma=(5, 0), order=0))
    sub1s2.image = im.fromarray(ndimage.gaussian_filter(np.asarray(sub1s2.image), sigma=(5, 0), order=0))

    for i in range(sub1s1.image.size[0]):
        for j in range(sub1s1.image.size[1]):
            if (sub1s1.image.getpixel((i,j)) != 0):
                print(sub1s1.image.getpixel((i,j)),sub1s2.image.getpixel((i,j)))
                input()


    '''
	