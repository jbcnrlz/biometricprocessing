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
        if len(splitedFileName) <= 2:
            splitedFileName = splitedFileName[0]
        elif len(splitedFileName) < 5 or ('_occluded_' in fileName):
            splitedFileName = '_'.join(splitedFileName[:2])
        else:
            splitedFileName = '_'.join(splitedFileName[:-4])
        if validationSize > 0 and valFiles[1].count(className) < trainFiles[1].count(className) and 'rotate' not in fileName:
            valFiles[0].append(f)
            valFiles[1].append(className)
            valFiles[2].append(splitedFileName)
            validationSize -= 1
        else:
            if len(sepData[1]) > 0:
                for fls in sepData[1][splitedFileName]:
                    trainFiles[0].append(fls)
                    trainFiles[1].append(className)

            trainFiles[0].append(f)
            trainFiles[1].append(className)


    return (trainFiles[0],valFiles[0])

def validateFolds(foldsGenerated,maxClasses,startingPoint):
    classQtCheck = [0] * maxClasses
    for f in foldsGenerated[0]:
        fileName = f.split(os.path.sep)[-1]
        cn = int(fileName.split('_')[0])
        classQtCheck[cn-startingPoint] += 1

    if classQtCheck.count(0) > 0:
        printString = ''
        while classQtCheck.count(0) > 0:
            printString += str(classQtCheck.index(0)) + ' '
            classQtCheck[classQtCheck.index(0)] = -1

        print('Classes sem representação na galeria: %s' % (printString))
        return False

    for f in foldsGenerated[1]:
        fileName = f.split(os.path.sep)[-1]
        fileName = fileName[fileName.index('_')+1:-4]
        for f1 in foldsGenerated[0]:
            if fileName in f1:
                print('Training and Validation contaminated')
                return False

    return True

def main():
    parser = argparse.ArgumentParser(description='Generate folder with training data')
    parser.add_argument('--pathBase', help='Path for faces', required=True)
    parser.add_argument('--outputPath', help='Path in which faces are going to be saved', required=True)
    parser.add_argument('--validationSize', help='Size of the validation data', required=True)
    parser.add_argument('--classQuantity', help='Quantity of classes', required=True, type=int)
    parser.add_argument('--startingPoint', help='Number of first class', required=False, type=int, default=1)
    args = parser.parse_args()

    folds = getFilesFromFolder(args.pathBase,args.validationSize)
    print('Validating dataset')
    if validateFolds(folds,args.classQuantity, args.startingPoint):
        print('Fold is valid')
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

    else:
        print('Fold separation is not valid')

if __name__ == '__main__':
    main()