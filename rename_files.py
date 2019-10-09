from helper.functions import getFilesInPath
import shutil, os

if __name__ == '__main__':
    files = getFilesInPath('eurecom_newdepth_dataset')
    subjects = {}
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        originalClassName = int(fileName.split('_')[1])

        newFileName = str(originalClassName) + '_' + fileName
        shutil.copy(f,os.path.join('eurecom_newdepth_dataset_renamed',newFileName))