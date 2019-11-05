from helper.functions import getDirectoriesInPath, getFilesInPath
import os, argparse, shutil

def main(pathDirs,fileInclude=['01']):
    filesReturn = []
    dirs = getDirectoriesInPath(pathDirs)
    for d in dirs:
        files = getFilesInPath(os.path.join(pathDirs,d))
        for f in files:
            fileName = f.split(os.path.sep)[-1].split('_')[-2]
            if fileName in fileInclude:
                filesReturn.append(f)


    return filesReturn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and extract Lock 3D database')
    parser.add_argument('--pathdatabase', help='Path for the database', required=True)
    parser.add_argument('--outputpathdatabase', help='Output path for the database', required=True)
    parser.add_argument('--filesInclude', help='Output path for the database', required=False, default='01')
    args = parser.parse_args()

    files = main(args.pathdatabase,args.filesInclude.split('_'))
    if not os.path.exists(args.outputpathdatabase):
        os.makedirs(args.outputpathdatabase)

    for f in files:
        folder = f.split(os.path.sep)[-2]
        fileName = f.split(os.path.sep)[-1]
        if not os.path.exists(os.path.join(args.outputpathdatabase,folder)):
            os.makedirs(os.path.join(args.outputpathdatabase,folder))

        shutil.copyfile(f,os.path.join(args.outputpathdatabase,folder,fileName))