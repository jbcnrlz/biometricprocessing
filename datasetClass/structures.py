from torch.utils.data import Dataset
from helper.functions import loadFoldFromFolders, getFilesInPath, readFeatureFile
import os, numpy as np, torch
from PIL import Image as im

def getSameFileFromFoldersFeatures(fileNameParent, siblist):
    fileNameParent = fileNameParent.split(os.path.sep)[-1]
    fileName = fileNameParent[fileNameParent.index('_')+1:fileNameParent.index('.')]

    for idxSib, s in enumerate(siblist):
        simFilename = s.split(os.path.sep)[-1]
        simFilename = simFilename[simFilename.index('_') + 1:simFilename.index('.')]
        if simFilename == fileName:
            return idxSib


def loadFeaturesFromText(pathFold,validationSize=0,transforms=None):

    dataFile = []
    for p in pathFold.split('__'):
        dataFile.append(readFeatureFile(p))

    if validationSize == 'auto':
        validationSize = int(len(dataFile[0][0]) / 10)
    trainFiles = [[], []]
    valFiles = [[], [], []]
    for idx,df in enumerate(dataFile[0][0]):
        siblist = [df] + [dataFile[idxItem][0][getSameFileFromFoldersFeatures(dataFile[0][2][idx],dataFile[idxItem][2])] for idxItem in range(1,len(dataFile))]

        if validationSize is None:
            trainFiles[0].append(siblist)
            trainFiles[1].append(dataFile[0][1][idx])

        else:
            fileName = dataFile[0][2][idx].split(os.path.sep)[-1].strip()
            fileId = '_'.join(fileName.split('_')[:-4])
            className = int(''.join([lt for lt in fileName.split('_')[0] if not lt.isalpha()]))
            if validationSize > 0 and (
                    className not in valFiles[1] or valFiles[1].count(dataFile[0][1][idx]) < trainFiles[1].count(dataFile[0][1][idx])) and 'rotate' not in fileName:
                valFiles[0].append(siblist)
                valFiles[1].append(dataFile[0][1][idx])
                valFiles[2].append(fileId)
                validationSize -= 1
            elif '_'.join(fileName.split('_')[:-4]) not in valFiles[2]:
                trainFiles[0].append(siblist)
                trainFiles[1].append(dataFile[0][1][idx])

    if int(min(trainFiles[1])) == 1:
        trainFiles[1] = [int(t) - 1 for t in trainFiles[1]]
        valFiles[1] = [int(t) - 1 for t in valFiles[1]]

    if validationSize is None:
        galDataLoader = SiameseFoldsFeatures(trainFiles[0], trainFiles[1], transforms)
        return galDataLoader
    else:
        galDataLoader = SiameseFoldsFeatures(trainFiles[0], trainFiles[1], transforms)
        proDataLoader = SiameseFoldsFeatures(valFiles[0], valFiles[1], transforms)
        return (galDataLoader, proDataLoader)

def loadFoldsDatasets(pathFolds,transforms=None):
    folds = loadFoldFromFolders(pathFolds)
    returnDataFolds = []
    for f in folds:
        if int(min(f[1])) == 1:
            f[1] = [int(t) - 1 for t in f[1]]
            f[3] = [int(t) - 1 for t in f[3]]

        galDataLoader = Folds(f[0],f[1],transforms)
        proDataLoader = Folds(f[2],f[3],transforms)
        returnDataFolds.append((galDataLoader,proDataLoader))
    return returnDataFolds

def loadFoldsDatasetsDepthDI(pathFolds,transforms=None,depthFolder=None,depthTransform=None):
    folds = loadFoldFromFolders(pathFolds)
    returnDataFolds = []
    for f in folds:
        if int(min(f[1])) == 1:
            f[1] = [int(t) - 1 for t in f[1]]
            f[3] = [int(t) - 1 for t in f[3]]

        galDataLoader = FoldsWithDepth(f[0],f[1],transforms,depthFolder=depthFolder,depthTransform=depthTransform)
        proDataLoader = FoldsWithDepth(f[2],f[3],transforms,depthFolder=depthFolder,depthTransform=depthTransform)
        returnDataFolds.append((galDataLoader,proDataLoader))
    return returnDataFolds


def getSameFileFromFolders(fileName,folders):
    fileName = fileName[fileName.index('_')+1:fileName.index('.')]
    returnPaths = []
    for files in folders:
        for fn in files:
            simFilename = fn.split(os.path.sep)[-1]
            simFilename = simFilename[simFilename.index('_')+1:simFilename.index('.')]
            if simFilename == fileName:
                returnPaths.append(fn)
                break

    return returnPaths

def loadSiameseDatasetFromFolder(pathFold,otherFolds,validationSize=0,transforms=None):
    files = getFilesInPath(pathFold)
    otherFoldsFiles = [getFilesInPath(of) for of in otherFolds]
    if validationSize == 'auto':
        validationSize = int(len(files) / 10)
    trainFiles = [[],[]]
    valFiles = [[],[],[]]
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        className = int(''.join([lt for lt in fileName.split('_')[0] if not lt.isalpha()]))
        siblings = getSameFileFromFolders(fileName,otherFoldsFiles)
        siblings.append(f)
        if validationSize is None:
            trainFiles[0].append(siblings)
            trainFiles[1].append(className)
        else:
            if validationSize > 0 and (className not in valFiles[1] or valFiles[1].count(className) < trainFiles[1].count(className)) and 'rotate' not in fileName:
                valFiles[0].append(siblings)
                valFiles[1].append(className)
                valFiles[2].append('_'.join(fileName.split('_')[:-4]))
                validationSize -= 1
            elif '_'.join(fileName.split('_')[:-4]) not in valFiles[2]:
                trainFiles[0].append(siblings)
                trainFiles[1].append(className)

    if int(min(trainFiles[1])) == 1:
        trainFiles[1] = [int(t) - 1 for t in trainFiles[1]]
        valFiles[1] = [int(t) -1 for t in valFiles[1]]

    if validationSize is None:
        galDataLoader = SiameseFolds(trainFiles[0],trainFiles[1],transforms)
        return galDataLoader
    else:
        galDataLoader = SiameseFolds(trainFiles[0],trainFiles[1],transforms)
        proDataLoader = SiameseFolds(valFiles[0],valFiles[1],transforms)
        return (galDataLoader,proDataLoader)

def loadFullSet(pathFiles,transforms=None):
    files = getFilesInPath(pathFiles)
    trainFiles = [[], []]
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        className = int(''.join([lt for lt in fileName.split('_')[0] if not lt.isalpha()]))
        trainFiles[0].append(f)
        trainFiles[1].append(className)
    return Folds(trainFiles[0],trainFiles[1],transforms)

def loadDatasetFromFolder(pathFold,validationSize=0,transforms=None,size=(100,100)):
    files = getFilesInPath(pathFold)
    if validationSize == 'auto':
        validationSize = int(len(files) / 10)
    trainFiles = [[],[]]
    valFiles = [[],[],[]]
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        className = int(''.join([lt for lt in fileName.split('_')[0] if not lt.isalpha()]))
        splitedFileName = fileName.split('_')
        if len(splitedFileName) < 5:
            splitedFileName = '_'.join(splitedFileName[:2])
        else:
            splitedFileName = '_'.join(splitedFileName[:-4])
        if validationSize > 0 and (className not in valFiles[1] or valFiles[1].count(className) < trainFiles[1].count(className)) and 'rotate' not in fileName:
            valFiles[0].append(f)
            valFiles[1].append(className)
            valFiles[2].append(splitedFileName)
            validationSize -= 1
        elif splitedFileName not in valFiles[2]:
            trainFiles[0].append(f)
            trainFiles[1].append(className)

    if int(min(trainFiles[1])) == 1:
        trainFiles[1] = [int(t) - 1 for t in trainFiles[1]]
        valFiles[1] = [int(t) -1 for t in valFiles[1]]

    galDataLoader = Folds(trainFiles[0],trainFiles[1],transforms)
    proDataLoader = Folds(valFiles[0],valFiles[1],transforms)
    return (galDataLoader,proDataLoader)

def loadFolder(pathFold,mode,transforms=None):
    files = getFilesInPath(pathFold)
    filesFold = [[],[]]
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        className = fileName.split('_')[0]
        filesFold[0].append(f)
        filesFold[1].append(className)

    if min(filesFold[1]) == 1:
        filesFold[1] = [int(t) - 1 for t in filesFold[1]]

    return Folds(filesFold[0],filesFold[1],transform=transforms,modeFile=mode)

def loadFolderDepthDI(pathFold,mode,transforms=None,depthFolder=None,depthTransform=None,filePaths=None):
    if filePaths is None:
        files = getFilesInPath(pathFold)
    else:
        files = filePaths
    filesFold = [[],[]]
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        className = fileName.split('_')[0]
        filesFold[0].append(f)
        filesFold[1].append(className)

    if min(filesFold[1]) == 1:
        filesFold[1] = [int(t) - 1 for t in filesFold[1]]

    return FoldsWithDepth(filesFold[0],filesFold[1],transform=transforms,modeFile=mode,depthFolder=depthFolder,depthTransform=depthTransform)


def pil_loader(path,mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            img = im.open(f)
            return img.convert(mode)
    except:
        print(path)
        print('erro')
        input()

def mat_loader(path,mode):
    import h5py
    arrays = {}
    image = None
    try:
        with h5py.File(path) as fPy:
            for k, v in fPy.items():
                arrays[k] = np.array(v)
    except:
        import scipy.io as sio
        arrays = sio.loadmat(path)

    image = arrays['defShape'].T
    return image

def npy_loader(path,mode):
    return np.load(path)

class SiameseFoldsFeatures(Dataset):
    def __init__(self, features, classes, transform=None, target_transform=None):
        self.classes = list(map(int,classes))
        self.samples = features
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):

        if self.target_transform is not None:
            self.classes[index] = self.target_transform(self.classes[index])
        #finalImageLoaded = np.array(finalImageLoaded)
        if self.transform is not None:
            for i in range(len(self.samples[index])):
                self.samples[index][i] = self.transform(np.array(self.samples[index][i]).reshape(1,-1,1).astype(np.float32))


        return self.samples[index], self.classes[index]


class SiameseFolds(Dataset):
    def __init__(self, files, classes, transform=None, target_transform=None, modeFile='auto'):
        self.classes = list(map(int,classes))
        self.samples = files
        self.transform = transform
        self.target_transform = target_transform
        self.modeFile = modeFile

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):
        finalImageLoaded = []
        for idx, fName in enumerate(self.samples[index]):
            '''
            if type(self.modeFile) is list:
                if self.modeFile[idx] == 'auto':
                    mode = 'RGBA' if fName[-3:].lower() == 'png' else 'RGB'
                else:
                    mode = self.modeFile
            else:
                if self.modeFile == 'auto':
                    mode = 'RGBA' if fName[-3:].lower() == 'png' else 'RGB'
                else:
                    mode = self.modeFile

            lImage = pil_loader(fName,mode)            
            '''
            lImage = pil_loader(fName, 'RGBA')
            finalImageLoaded.append(lImage)

        if self.target_transform is not None:
            self.classes[index] = self.target_transform(self.classes[index])
        #finalImageLoaded = np.array(finalImageLoaded)
        if self.transform is not None:
            for i in range(len(finalImageLoaded)):
                finalImageLoaded[i] = self.transform(finalImageLoaded[i]) / 255


        return finalImageLoaded, self.classes[index]

class Folds(Dataset):

    def __init__(self, files, classes, transform=None, target_transform=None, modeFile='auto'):
        self.classes = list(map(int,classes))
        self.samples = files

        if files[0][-3:] == 'mat':
            self.loader = mat_loader
        elif files[0][-3:] == 'npy':
            self.loader = npy_loader
        else:
            self.loader = pil_loader

        self.transform = transform
        self.target_transform = target_transform
        self.modeFile = modeFile

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):
        if self.modeFile == 'auto':
            mode = 'RGBA' if self.samples[index][-3:].lower() == 'png' else 'RGB'
        else:
            mode = self.modeFile
        sample = self.loader(self.samples[index],mode)
        target = self.classes[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(self.classes[index])

        return sample.float() / 255, target

class FoldsWithDepth(Dataset):

    def __init__(self, files, classes, transform=None, target_transform=None, modeFile='auto',depthFolder=None,depthTransform=None):
        self.classes = list(map(int,classes))
        self.samples = files
        self.depthFolder = depthFolder
        self.depthTransform = depthTransform
        if files[0][-3:] == 'mat':
            self.loader = mat_loader
        elif files[0][-3:] == 'npy':
            self.loader = npy_loader
        else:
            self.loader = pil_loader

        self.transform = transform
        self.target_transform = target_transform
        self.modeFile = modeFile

    def getDepthFileName(self,fileNameDI):
        if 'frgc' in fileNameDI:
            fileName = fileNameDI.split(os.path.sep)[-1].split('.')[0] + '.jpeg'
        elif 'eurecom' in fileNameDI:
            if 'newdepth' in fileNameDI:
                fileName = fileNameDI.split(os.path.sep)[-1].split('.')[0] + '.bmp'
            else:
                fileName = fileNameDI.split(os.path.sep)[-1].split('.')[0] + '_newdepth.bmp'
        elif 'iiitd' in fileNameDI:
            fileName = fileNameDI.replace('_depthnocolor','').split(os.path.sep)[-1]
            if 'newdepth' not in fileName:
                fileName = fileName[:-4] + '_newdepth.bmp'
            else:
                fileName = fileName[:-4] + '.bmp'
        elif 'bosphorus' in fileNameDI:
            fileName = fileNameDI.split(os.path.sep)[-1].split('.')[0] + '.bmp'
            fileName = fileName.split('_')
            fileName[0] = '%03d' % int(fileName[0])
            fileName = '_'.join(fileName)
        elif 'micc' in fileNameDI:
            fileName = fileNameDI.split(os.path.sep)[-1].split('.')[0] + '.bmp'
            if 'ld' in fileName and 'newdepth' not in fileName:
                fileName = fileName[:-6] + 'newdepth_ld.bmp'
        else:
            fileName = fileNameDI.split(os.path.sep)[-1].split('.')[0] + '.jpeg'

        return fileName

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):
        if self.modeFile == 'auto':
            mode = 'RGBA' if self.samples[index][-3:].lower() == 'png' else 'RGB'
        else:
            mode = self.modeFile
        sample = self.loader(self.samples[index],mode)
        fileName = self.getDepthFileName(self.samples[index])
        sampleDepth = self.loader(os.path.join(self.depthFolder,fileName),'RGB')
        if self.depthTransform is not None:
            sampleDepth = self.depthTransform(sampleDepth)

        target = self.classes[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(self.classes[index])

        return (sample.float() / 255, sampleDepth.float() / 255), target