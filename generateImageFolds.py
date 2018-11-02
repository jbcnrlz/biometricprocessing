from helper.functions import generateData, generateFoldsOfData
import argparse, shutil, os

def copyFilesToFolder(datas,folder):
    for d in datas:
        fName = d.split(os.path.sep)[-1]
        shutil.copy(d,os.path.join(folder,fName))

def validateFold(foldedData):
    for fNumber, fData in enumerate(foldedData):
        probe = fData[2]
        galle = fData[0]
        for p in probe:
            cPFile = p.split(os.path.sep)[-1]
            if cPFile.find('newdepth') == -1:
                cPFile = cPFile[:-3]
            else:
                cPFile = cPFile[:cPFile.find('newdepth')+len('newdepth')]
            for g in galle:
                gFile = g.split(os.path.sep)[-1]
                if gFile.find('newdepth') == -1:
                    gFile = gFile[:-3]
                else:
                    gFile = gFile[:gFile.find('newdepth') + len('newdepth')]
                if (cPFile == gFile):
                    return False

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate image folds')
    parser.add_argument('-p', '--pathBase',default='generated_images_lbp_frgc',help='Path for faces', required=False)
    parser.add_argument('-f', '--folds', type=int, default=10, help='Fold quantity', required=False)
    parser.add_argument('--folder', default='fold_folder', help='Fold quantity', required=False)
    args = parser.parse_args()

    imageData, classesData = generateData(args.pathBase)
    folds = generateFoldsOfData(args.folds, imageData, classesData)

    if os.path.exists(args.folder):
        shutil.rmtree(args.folder)

    os.makedirs(args.folder)

    if validateFold(folds):

        for f, datas in enumerate(folds):
            foldProbe   = datas[2]
            foldGallery = datas[0]
            os.makedirs(os.path.join(args.folder,str(f+1)))
            os.makedirs(os.path.join(args.folder, str(f + 1),'gallery'))
            os.makedirs(os.path.join(args.folder, str(f + 1),'probe'))
            copyFilesToFolder(foldProbe,os.path.join(args.folder, str(f + 1),'probe'))
            copyFilesToFolder(foldGallery, os.path.join(args.folder, str(f + 1), 'gallery'))

    else:
        print('Not valid fold division')