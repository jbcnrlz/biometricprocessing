from helper.functions import getFilesInPath
import shutil, os

if __name__ == '__main__':
    files = getFilesInPath('rgb_eurecom')
    subjects = {}
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        originalClassName = fileName.split('_')

        newFileName = originalClassName[1] + '_' + originalClassName[2] + '_' + '_'.join(originalClassName[0:])
        shutil.copy(f,os.path.join('rgb_eurecom_renamed',newFileName))