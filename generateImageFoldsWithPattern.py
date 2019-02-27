from helper.functions import generateData, generateExperimentDataPattern, loadPatternFromFiles
import argparse, shutil, os

def copyFilesToFolder(datas,folder):
    for d in datas:
        fName = d.split(os.path.sep)[-1]
        shutil.copy(d,os.path.join(folder,fName))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experiment')
    parser.add_argument('-p', '--pathBase',default='generated_images_lbp_frgc',help='Path for faces', required=False)
    parser.add_argument('--folder', default='fold_folder', help='Fold for folders', required=False)
    parser.add_argument('--patternProbe', help='Pattern for probe files', required=False)
    parser.add_argument('--patternGallery', help='Pattern for gallery files', required=False, default=None)
    parser.add_argument('--filePattern', help='File with patterns', required=False, default=None)
    parser.add_argument('--extension', help='Extension from files', required=False, default='png')
    args = parser.parse_args()

    imageData, classesData = generateData(args.pathBase,extension=args.extension)

    if os.path.exists(args.folder):
        shutil.rmtree(args.folder)

    os.makedirs(args.folder)

    if args.filePattern is not None:
        folds = loadPatternFromFiles(args.filePattern)
        for i,f in enumerate(folds):
            os.makedirs(os.path.join(args.folder, str(i+1), 'gallery'))
            os.makedirs(os.path.join(args.folder, str(i+1), 'probe'))
            experimentBase = generateExperimentDataPattern(imageData, classesData, f[1].split('__'), f[0].split('__'))
            copyFilesToFolder(experimentBase[2], os.path.join(args.folder, str(i+1), 'probe'))
            copyFilesToFolder(experimentBase[0], os.path.join(args.folder, str(i+1), 'gallery'))
    else:
        galleryPattern = args.patternGallery if args.patternGallery is None else args.patternGallery.split('__')
        experimentBase = generateExperimentDataPattern(imageData, classesData,args.patternProbe.split('__'),galleryPattern)

        os.makedirs(os.path.join(args.folder, 'gallery'))
        os.makedirs(os.path.join(args.folder, 'probe'))

        copyFilesToFolder(experimentBase[2], os.path.join(args.folder, 'probe'))
        copyFilesToFolder(experimentBase[0], os.path.join(args.folder, 'gallery'))
