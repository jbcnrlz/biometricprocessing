from helper.functions import getFilesInPath, getDirectoriesInPath
import argparse, os, shutil

def main():
    parser = argparse.ArgumentParser(description='Generate and rename new depth folder')
    parser.add_argument('--pathBase',help='Path for MICC Dataset', required=True)
    parser.add_argument('--DIFolder', help='Folder with Descriptor Images', required=True)
    parser.add_argument('--depthFolder', help='Folder to generate', required=True)
    parser.add_argument('--extension', help='Extensions from file', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.depthFolder):
        os.makedirs(args.depthFolder)

    disFiles = getFilesInPath(args.DIFolder,fullPath=False)
    for scanType in getDirectoriesInPath(args.pathBase):
        for subject in getDirectoriesInPath(os.path.join(args.pathBase,scanType,'processed')):
            if 'kinect' in scanType:
                for file in getFilesInPath(os.path.join(args.pathBase,scanType,'processed',subject)):
                    fileName = file.split(os.path.sep)[-1]
                    extension = fileName.split('.')[-1]
                    if extension == args.extension:
                        copiedFilename = subject + '_' + fileName.split('.')[0] + '_ld.' + extension
                        shutil.copyfile(file,os.path.join(args.depthFolder,copiedFilename))
            else:
                files = getFilesInPath(os.path.join(args.pathBase,scanType,'processed',subject))
                for file in files:
                    if 'cropped' in file and file[-3:] == args.extension:
                        fileName = file.split(os.path.sep)[-1]
                        for d in disFiles:                            
                            orrFile = d.split('_')
                            if int(subject) == int(orrFile[0]):
                                fileCompost = '_'.join(orrFile[2:-1])
                                if fileCompost in fileName:
                                    if 'rotate' in file:
                                        fileRotate = fileName.split('_')
                                        fileNameDepthCopy = '%s_%s_%s_cropped_rotate_%s_%s_newdepth_hd.%s' % (orrFile[0],orrFile[1],orrFile[2],fileRotate[3],fileRotate[4],args.extension)
                                        shutil.copyfile(file,os.path.join(args.depthFolder,fileNameDepthCopy))
                                    else:
                                        shutil.copyfile(file,os.path.join(args.depthFolder,'_'.join(orrFile)[:-3]+args.extension))
                            


if __name__ == '__main__':
    main()