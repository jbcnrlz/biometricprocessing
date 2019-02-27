from torch.utils.data import Dataset
from helper.functions import loadFoldFromFolders, getFilesInPath
import os, numpy as np
from PIL import Image as im

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

def loadDatasetFromFolder(pathFold,validationSize = 0,transforms=None):
    files = getFilesInPath(pathFold)
    if validationSize == 'auto':
        validationSize = int(len(files) / 10)
    trainFiles = [[],[]]
    valFiles = [[],[],[]]
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        className = fileName.split('_')[0]
        if validationSize > 0 and (className not in valFiles[1] or valFiles[1].count(className) < trainFiles[1].count(className)) and 'rotate' not in fileName:
            valFiles[0].append(f)
            valFiles[1].append(className)
            valFiles[2].append('_'.join(fileName.split('_')[:-4]))
            validationSize -= 1
        elif '_'.join(fileName.split('_')[:-4]) not in valFiles[2]:
            trainFiles[0].append(f)
            trainFiles[1].append(className)

    if int(min(trainFiles[1])) == 1:
        trainFiles[1] = [int(t) - 1 for t in trainFiles[1]]
        valFiles[1] = [int(t) -1 for t in valFiles[1]]

    galDataLoader = Folds(trainFiles[0],trainFiles[1],transforms)
    proDataLoader = Folds(valFiles[0],valFiles[1],transforms)
    return (galDataLoader,proDataLoader)

def loadFolder(pathFold,transforms=None):
    files = getFilesInPath(pathFold)
    filesFold = [[],[]]
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        className = fileName.split('_')[0]
        filesFold[0].append(f)
        filesFold[1].append(className)

    if min(filesFold[1]) == 1:
        filesFold[1] = [int(t) - 1 for t in filesFold[1]]

    return Folds(filesFold[0],filesFold[1],transforms)

def pil_loader(path,mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = im.open(f)
        return img.convert(mode)

class Folds(Dataset):

    def __init__(self, files, classes, transform=None, target_transform=None):
        self.classes = list(map(int,classes))
        self.samples = files

        self.loader = pil_loader

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
        mode = 'RGBA' if self.samples[index][-3:].lower() == 'png' else 'RGB'
        sample = np.array(self.loader(self.samples[index],mode)) / 255
        target = self.classes[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(self.classes[index])

        return sample.float(), target