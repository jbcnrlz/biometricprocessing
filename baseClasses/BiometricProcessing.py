from helper.functions import outputObj
from multiprocessing import Pool, Process, Pipe
import traceback, datetime
from models.models import *
from models.engine_creation import *

class BiometricProcessing:

    methodName = None
    databases = None
    __preProcessingSteps = None
    modelDatabase = None

    def __init__(self):
        self.modelDatabase = self.getModel()

    def saveFeature(self,template):
        fts = ' '.join([str(a) for a in template.features])
        dts = ss.query(Template).filter(Template.amostra==template.rawRepr,Template.metodo==self.modelDatabase).first()
        if (not dts):
            dts = Template(amostra=template.rawRepr,caracteristica=fts,data_extracao=datetime.datetime.now(),sujeito=template.modelDatabase,metodo=self.modelDatabase)
            ss.add(dts)            
        else:
            dts.caracteristica=fts
            dts.data_extracao=datetime.datetime.now()

        ss.commit()        
        ss.flush()

    def getModel(self):
        if ss is None:
            return None
        else:
            dts = ss.query(Metodo).filter(Metodo.nome==self.methodName).first()
            if (not dts):
                dts = Metodo(nome=self.methodName)
                ss.add(dts)
                ss.commit()

            return dts

    def getPreProcessingSteps(self):
        return self.__preProcessingSteps

    def setPreProcessingSteps(self,value):
        if self.__preProcessingSteps == None:
            self.__preProcessingSteps = [value]
        else:
            self.__preProcessingSteps.append(value)

    preProcessingSteps = property(getPreProcessingSteps,setPreProcessingSteps)

    def applyPreProcessing(self,template,verbose=True,forceImage=True):
        template = self.setupTemplate(template)
        if forceImage or not template.existsPreProcessingFile():
            if verbose:
                print(str(template.rawRepr))

            try:
                for p in self.preProcessingSteps:
                    if verbose:
                        print(p)
                    template = p.doPreProcessing(template)
            except Exception as e:
                traceback.print_exc()
            template = self.cleanupTemplate(template)

    def preProcessing(self,verbose=False,paralelCalling=True,procs=10):
        if paralelCalling:
            poolCalling = Pool(processes=procs)
            for database in self.databases:
                poolCalling.map(unwrap_self_f,zip([self]*len(database.templates), database.templates))
        else:
            for database in self.databases:
                for template in database.templates:                
                    self.applyPreProcessing(template,verbose)

    def featureExtraction(self):
        pass

    def matcher(self):
        pass

    def setupTemplate(self,template):
        return template

    def cleanupTemplate(self,template):
        return template

    def feedProbilityTemplate(self,classNumber,probabilities):
        template = self.databases[1].getTemplateFromSubject(classNumber).pop()
        template.probability = probabilities
        print(template.itemClass)
        print(template.probability)
        input()


    def getFullProcessingNumber(self):
        total = len(self.databases)
        for d in self.databases:
            total += len(d.templates)

        return total

    def doFeatureExtraction(self, parameters, verbrose=True):
        return self.localcall(parameters)

    def localcall(self, parameters):
        pass

def unwrap_self_f(arg, **kwarg):
    return BiometricProcessing.applyPreProcessing(*arg, **kwarg)

def unwrap_self_f_feature(arg, **kwarg):
    return BiometricProcessing.doFeatureExtraction(*arg, **kwarg)
