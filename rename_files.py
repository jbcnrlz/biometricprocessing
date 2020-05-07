from helper.functions import getFilesInPath
import shutil, os

if __name__ == '__main__':
    files = getFilesInPath('bosphorus_depth')
    subjects = {}
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        originalClassName = fileName.split('_')

        newFileName = originalClassName[0][2:] + '_' + fileName
        shutil.copy(f,os.path.join('bosphorus_depth_renamed',newFileName))