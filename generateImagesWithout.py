from helper.functions import getFilesInPath
import os, sys, shutil,re, argparse

def getNewDatabase(pathMain,patterns=[]):
    returnFilesDirectory = []
    allFiles = getFilesInPath(pathMain)
    for a in allFiles:
        fileExam = a.split(os.path.sep)[-1]
        imageType = fileExam.split('_')[2]
        if imageType in patterns:
            returnFilesDirectory.append(a)
    return returnFilesDirectory

def generateNewDatase(newPath,files):
    if (os.path.exists(newPath)):
        shutil.rmtree(newPath)

    os.makedirs(newPath)
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        shutil.copy(f,os.path.join(newPath,fileName))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Separate files')
    parser.add_argument('-p', '--pathBase',help='Path for faces', required=True)
    parser.add_argument('-n', '--newFolder',help='New Generated Folder', required=True)
    parser.add_argument('--pattern', help='Accepted Pattern', required=True)
    args = parser.parse_args()

    files = getNewDatabase(args.pathBase,args.pattern.split('_'))
    generateNewDatase(args.newFolder,files)