from helper.functions import getFilesInPath, getDirectoriesInPath
import os, sys, shutil,re
from PIL import Image as im

def getNewDatabase(pathMain,patterns=[]):
    allDirs = getDirectoriesInPath(pathMain)
    returnFilesDirectory = {}
    for d in allDirs:
        allFiles = getFilesInPath(os.path.join(pathMain,d))
        for a in allFiles:
            fileExam = a.split(os.path.sep)[-1]
            for p in patterns:
                if re.match(p,fileExam):
                    if d in returnFilesDirectory.keys():
                        returnFilesDirectory[d].append(a)
                    else:
                        returnFilesDirectory[d] = [a]

                    break
    return returnFilesDirectory

def generateNewDatase(newPath,files,rotateDegress=False):
    currDir = 1
    for f in files:
        os.makedirs(os.path.join(newPath,str(currDir)))
        for d in files[f]:
            fullOldPath = os.path.join(d)
            fileName = d.split(os.path.sep)[-1]
            fullNewsPath = os.path.join(newPath,str(currDir),fileName)
            shutil.copy(fullOldPath,fullNewsPath)
            if rotateDegress:

                iRotate = im.open(fullNewsPath)
                iRotate.rotate(rotateDegress).save(fullNewsPath)

        currDir += 1



if __name__ == '__main__':
    if os.path.exists(sys.argv[2]):
        shutil.rmtree(sys.argv[2])

    degreesRotate = False if len(sys.argv) < 5 else float(sys.argv[4])

    os.makedirs(sys.argv[2])

    patterns = sys.argv[3].split('__')

    databaseOldFiles = getNewDatabase(sys.argv[1],patterns)
    generateNewDatase(sys.argv[2],databaseOldFiles,degreesRotate)


