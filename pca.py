from FRGCTemplate import *
from baseClasses.BiometricProcessing import *
from BosphorusTemplate import *
from sklearn.decomposition import PCA

class PCAImpl(BiometricProcessing):

    def __init__(self,database,folderSave):
        self.folderImages = folderSave
        self.databases = database

    def setupTemplate(self,template):
        return template

    def cleanupTemplate(self,template):
        if (type(template) is not FRGCTemplate) and (type(template) is not BosphorusTemplate) and (len(template.image) > 0):
            template.layersChar = np.zeros((len(template.image), len(template.image[0]), 4))
            template.image = im.fromarray(np.array(template.image, dtype=np.uint8))
            template.image = template.image.rotate(-180)
            template.save(True)
        return template

    def featureExtraction(self):
        for database in range(len(self.databases)):
            for template in self.databases[database].templates:
                if type(template.image) is np.ndarray:
                    template.image = im.fromarray(template.image)
                invImage = PIL.ImageOps.invert(template.image)
                pathImage = template.rawRepr.split(os.path.sep)[-1]
                if template.folderTemplate is not None:
                    invImage.save(os.path.join(self.folderImages,str(template.itemClass)+'_'+template.folderTemplate+'_'+pathImage))
                else:
                    invImage.save(os.path.join(self.folderImages, str(
                        template.itemClass) + '_' + pathImage))
        '''
        pca = PCA(n_components=2500,svd_solver='full')
        print("Iniciando feature extraction")        
        for database in range(len(self.databases)):
            newDb = []
            for template in self.databases[database].templates:
                newDb.append(np.array(template.image).flatten())

            newDb = np.array(newDb)
            pca.fit(newDb)
            newDb = pca.transform(newDb)
            print('nonono')
        '''