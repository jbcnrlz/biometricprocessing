import argparse,shutil, os
from helper.functions import getFilesInPath, getDirectoriesInPath

def main():
    parser = argparse.ArgumentParser(description='Extract images from dataset lock')
    parser.add_argument('--pathBase',help='Path for faces', required=True)
    parser.add_argument('--pathTo', help='Path for faces', required=True)
    args = parser.parse_args()

    afs = getDirectoriesInPath(args.pathBase)

    for dr in afs:
        files = getFilesInPath(os.path.join(args.pathBase,dr))
        for fl in files:
            pathSplited = fl.split(os.path.sep)
            if pathSplited[-1][-3:] == 'bmp':
                if not os.path.exists(os.path.join(args.pathTo,pathSplited[-2])):
                    os.makedirs(os.path.join(args.pathTo,pathSplited[-2]))

                shutil.copyfile(fl,os.path.join(args.pathTo,pathSplited[-2],pathSplited[-1]))


if __name__=='__main__':
    main()