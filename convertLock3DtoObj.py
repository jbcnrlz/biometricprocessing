import argparse, shutil, os, cv2
from helper.functions import getFilesInPath, getDirectoriesInPath, outputObj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates 3DOBJ to Lock3D')
    parser.add_argument('--pathBase', help='Path to Lock3D', required=True)
    parser.add_argument('--pathOBJ',  help='Path to save OBJ data', required=True)
    args = parser.parse_args()

    if os.path.exists(args.pathOBJ):
        shutil.rmtree(args.pathOBJ)

    os.makedirs(args.pathOBJ)

    folders = getDirectoriesInPath(args.pathBase)
    for f in folders:
        folderName = f.split(os.path.sep)[-1]
        className = folderName.split('_')[0]
        filesInFolder = getFilesInPath(os.path.join(args.pathBase,f))
        folderOBJ = os.path.join(args.pathOBJ,className)
        for ff in filesInFolder:
            dImg = cv2.imread(ff,-1)
            fileName = ff.split(os.path.sep)[-1]
            fileName = fileName.split('.')[0]
            cloudRepr = [[i,j,dImg[i,j]] for i in range(dImg.shape[0]) for j in range(dImg.shape[1])]

            if not os.path.exists(folderOBJ):
                os.makedirs(folderOBJ)

            outputObj(cloudRepr,os.path.join(folderOBJ,folderName+'_'+fileName+'.obj'))