from helper.functions import generateData, generateExperimentDataPattern
import argparse, shutil, os

def copyFilesToFolder(datas,folder):
    for d in datas:
        fName = d.split(os.path.sep)[-1]
        shutil.copy(d,os.path.join(folder,fName))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experiment')
    parser.add_argument('-p', '--pathBase',default='generated_images_lbp_frgc',help='Path for faces', required=False)
    parser.add_argument('--folder', default='fold_folder', help='Fold for folders', required=False)
    parser.add_argument('--patternProbe', help='Pattern for probe files', required=True)
    parser.add_argument('--patternGallery', help='Pattern for gallery files', required=False, default=None)
    args = parser.parse_args()

    galleryPattern = args.patternGallery if args.patternGallery is None else args.patternGallery.split('__')

    imageData, classesData = generateData(args.pathBase)
    experimentBase = generateExperimentDataPattern(imageData, classesData,args.patternProbe.split('__'),galleryPattern)

    if os.path.exists(args.folder):
        shutil.rmtree(args.folder)

    os.makedirs(args.folder)
    os.makedirs(os.path.join(args.folder, 'gallery'))
    os.makedirs(os.path.join(args.folder, 'probe'))

    copyFilesToFolder(experimentBase[2], os.path.join(args.folder, 'probe'))
    copyFilesToFolder(experimentBase[0], os.path.join(args.folder, 'gallery'))
