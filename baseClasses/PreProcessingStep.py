class PreProcessingStep:

    def __repr__(self):
        return 'Fazendo passo ' + self.__class__.__name__

    def doPreProcessing(self,template):
        return template