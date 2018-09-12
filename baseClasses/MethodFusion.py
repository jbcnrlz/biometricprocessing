from baseClasses.DatabaseProcessingUtility import *
import numpy as np
from sklearn import svm

class MethodFusion:

    __galleries = None

    def getGalleries(self):
        return self.__galleries

    def setGalleries(self,value):
        if self.__galleries == None:
            self.__galleries = [value]
        else:
            self.__galleries.append(value)

    galleries = property(getGalleries,setGalleries)

    __probes = None

    def getProbes(self):
        return self.__probes

    def setProbes(self,value):
        if self.__probes == None:
            self.__probes = [value]
        else:
            self.__probes.append(value)
            
    probes = property(getProbes,setProbes)

    def doFusion(self):
        probeAndGallery = [self.__probes,self.__galleries]
        for p in probeAndGallery:
            for i in range(1,len(p)):
                for j in range(len(p[0].templates)):
                    p[0].templates[j].features += p[i].templates[j].features

    def matcher(self):
        chutes = np.zeros(52)
        gal = self.__galleries[0].getDatabaseRepresentation()
        probe = self.__probes[0].getDatabaseRepresentation()

        tuned_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}

        
        for g in tuned_parameters['gamma']:
            for c in tuned_parameters['C']:
                clf = svm.SVC(gamma=g, C=c)
                clf.fit(np.array(gal[0]),np.array(gal[1]))
                resultados = [0,0]        
                for itemNumber in range(len(probe[0])):
                    classAchada = clf.predict(probe[0][itemNumber]).tolist().pop()
                    chutes[classAchada - 1] += 1
                    resultados[classAchada == probe[1][itemNumber]] += 1

                print(str(g) + ' ' + str(c))
                print(resultados)
                input()

        return []
        #return resultados