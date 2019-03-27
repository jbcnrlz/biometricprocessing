from helper.functions import getFilesInPath
import shutil, os

if __name__ == '__main__':
    filesRename = getFilesInPath('generated_images_bu_wfunc_3layers')
    for f in filesRename:
        fName = f.split(os.path.sep)[-1].split('_')
        fName = str(458 + int(fName[0])) + '_' + '_'.join(fName[1:])
        shutil.copy(f,os.path.join('fused_dataset',fName))
    print('oi')