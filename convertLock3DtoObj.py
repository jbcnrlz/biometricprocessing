import argparse, shutil, os, cv2
from helper.functions import getFilesInPath, getDirectoriesInPath, outputObj

def findLandmarkFile(currFile,lands):
    fileId = currFile.split('DEPTH')[0]
    for l in lands:
        if fileId in l:
            return l

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates 3DOBJ to Lock3D')
    parser.add_argument('--pathBase', help='Path to Lock3D', required=True)
    parser.add_argument('--pathOBJ',  help='Path to save OBJ data', required=True)
    parser.add_argument('--landmarks', help='Landmarks for files', default=None, required=False)
    args = parser.parse_args()

    if os.path.exists(args.pathOBJ):
        shutil.rmtree(args.pathOBJ)

    os.makedirs(args.pathOBJ)
    landMarksFiles = None if args.landmarks is None else getFilesInPath(args.landmarks)
    folders = getDirectoriesInPath(args.pathBase)
    for f in folders:
        print("Fixing folder %s" % (f))
        folderName = f.split(os.path.sep)[-1]
        className = folderName.split('_')[0]
        filesInFolder = getFilesInPath(os.path.join(args.pathBase,f))
        folderOBJ = os.path.join(args.pathOBJ,className)
        if not os.path.exists(folderOBJ):
            os.makedirs(folderOBJ)

        for ff in filesInFolder:
            fileName = ff.split(os.path.sep)[-1]
            shutil.copy(ff,os.path.join(folderOBJ,folderName+'_'+fileName))

            if landMarksFiles is not None:
                lFile = findLandmarkFile(folderName,landMarksFiles)
                shutil.copy(lFile,os.path.join(os.path.join(folderOBJ,folderName+'.txt')))