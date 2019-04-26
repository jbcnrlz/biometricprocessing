from helper.functions import getFilesInPath
import shutil, os

if __name__ == '__main__':
    files = getFilesInPath('generated_images_frgc_angle')
    subjects = {}
    for f in files:
        fileName = f.split(os.path.sep)[-1]
        originalClassName = int(fileName.split('_')[0])
        if originalClassName not in subjects.keys():
            subjects[originalClassName] = len(subjects) + 1

        newFileName = str(subjects[originalClassName]) + '_' + '_'.join(fileName.split('_')[1:])
        shutil.copy(f,os.path.join('generated_images_frgc_angle_renamed',newFileName))