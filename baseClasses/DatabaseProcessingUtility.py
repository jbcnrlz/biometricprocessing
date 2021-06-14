from sklearn.decomposition import PCA
import numpy as np

class DatabaseProcessingUtility:

    templates = None
    databasePath = None
    modelDatabase = None

    def getModel(self):
        pass

    def __init__(self,path):
        self.extensionFile = {'Range': 'jpg', 'NewDepth': 'jpeg', 'Depth': 'obj', 'Matlab' : 'mat', 'VRML' : 'off', 'DepthPng' : 'png','NewDepthBMP' : 'bmp','ply' : 'ply'}
        self.databasePath = path
        self.templates = []
        #self.modelDatabase = self.getModel()

    def feedTemplates(self):
        pass

    def getDatabaseRepresentation(self):
        pass

    def generateDatabaseFile(self,path):
        f = open(path,'w')
        f.write(str(len(self.templates))+' '+str(len(self.templates[0].features) + 1) + '\n')
        for t in self.templates:
            f.write(' '.join([str(e) for e in t.features]) + ' ' + str(t.itemClass - 1) + '\n')
        f.close()

    def generateListFromTemplates(self):
        featuresList = []
        for t in self.templates:
            featuresList.append([e for e in t.features])

        return np.array(featuresList)

    def applyPCA(self,components):
        pca_model = PCA(n_components=components)
        currData = self.generateListFromTemplates().T
        pca_model.fit_transform(currData)
        pcaData = pca_model.components_.T.tolist()
        for i in range(len(self.templates)):
            self.templates[i].features = pcaData[i]

