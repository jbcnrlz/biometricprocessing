from helper.functions import getFilesInPath
import shutil, os

if __name__ == '__main__':
    files = getFilesInPath('eurecom_newdepth_dataset')
    subjects = {}
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        originalClassName = fileName.split('_')

        newFileName = originalClassName[0] + '_' + originalClassName[3] + '_' + '_'.join(originalClassName[1:])
        shutil.copy(f,os.path.join('eurecom_newdepth_dataset_renamed',newFileName))