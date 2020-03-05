from helper.functions import getFilesInPath, getDirectoriesInPath
import argparse, os, shutil

def main():
    parser = argparse.ArgumentParser(description='Generate folder to train RGB')
    parser.add_argument('--pathOriginalData',help='Path for "Original data"', required=True)
    parser.add_argument('--pathRGBData',help='Path for RGB Faces', required=True)
    parser.add_argument('--pathForFolderToGenerate', help='Path with the folder to generate RGB Data', required=True)
    args = parser.parse_args()

    if os.path.exists(args.pathForFolderToGenerate):
        shutil.rmtree(args.pathForFolderToGenerate)

    os.makedirs(args.pathForFolderToGenerate)

    dirsFile = getDirectoriesInPath(args.pathOriginalData)
    conversionDir = {}
    for d in dirsFile:
        files = getFilesInPath(os.path.join(args.pathOriginalData,d))
        fileName = files[0].split(os.path.sep)[-1]
        originalClass = fileName.split('d')[0]
        conversionDir[originalClass] = d

    dirsFile = getDirectoriesInPath(args.pathRGBData)
    for d in dirsFile:
        files = getFilesInPath(os.path.join(args.pathRGBData,d))
        for f in files:
            fileName = f.split(os.path.sep)[-1]
            if 'rgb' in fileName:
                oldClass = fileName.split('d')[0]
                shutil.copyfile(f,os.path.join(args.pathForFolderToGenerate,conversionDir[oldClass] + '_' + fileName))

if __name__ == '__main__':
    main()