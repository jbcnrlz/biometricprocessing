import argparse, os, shutil
from helper.functions import getFilesInPath

def separateOriginalData(files):
    separatedData = [[],{}]
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        if 'rotate' in f:
            splitedFileName = fileName.split('_')
            if len(splitedFileName) < 5:
                splitedFileName = '_'.join(splitedFileName[:2])
            else:
                splitedFileName = '_'.join(splitedFileName[:-4])

            if splitedFileName in separatedData[1].keys():
                separatedData[1][splitedFileName].append(f)
            else:
                separatedData[1][splitedFileName] = [f]
        else:
            separatedData[0].append(f)
    return separatedData

def getFilesFromFolder(pathFold,validationSize='auto'):
    files = getFilesInPath(pathFold)
    sepData = separateOriginalData(files)
    if validationSize == 'auto':
        validationSize = int(len(files) / 10)
    trainFiles = [[],[]]
    valFiles = [[],[],[]]
    for f in sepData[0]:
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
            for fls in sepData[1][splitedFileName]:
                trainFiles[0].append(fls)
                trainFiles[1].append(className)

            trainFiles[0].append(f)
            trainFiles[1].append(className)


    return (trainFiles[0],valFiles[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate folder with training data')
    parser.add_argument('--pathBase', help='Path for faces', required=True)
    parser.add_argument('--outputPath', help='Path in which faces are going to be saved', required=True)
    parser.add_argument('--validationSize', help='Size of the validation data', required=True, type=int)
    args = parser.parse_args()

    folds = getFilesFromFolder(args.pathBase,args.validationSize)

    if os.path.exists(args.outputPath):
        shutil.rmtree(args.outputPath)

    os.makedirs(args.outputPath)
    os.makedirs(os.path.join(args.outputPath,'1','gallery'))
    os.makedirs(os.path.join(args.outputPath, '1', 'probe'))

    foldersName = ['gallery','probe']

    for i, fold in enumerate(folds):
        for t in fold:
            fileName = t.split(os.path.sep)[-1]
            shutil.copy(t,os.path.join(args.outputPath,'1',foldersName[i],fileName))