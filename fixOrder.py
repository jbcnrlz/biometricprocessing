from helper.functions import getFilesInPath
import argparse, os, shutil

def main():
    parser = argparse.ArgumentParser(description='Generate and rename new depth folder')
    parser.add_argument('--DIFolder', help='Folder with Descriptor Images', required=True)
    parser.add_argument('--rightDIFolder', help='Folder with Descriptor Images', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.rightDIFolder):
        os.makedirs(args.rightDIFolder)

    dis = getFilesInPath(args.DIFolder)
    subjectFiles = {}
    for d in dis:
        if 'ld' in d or 'rotate' in d:
            if 'ld' in d:
                fileName = d.split(os.path.sep)[-1]
                shutil.copyfile(d,os.path.join(args.rightDIFolder,fileName) )
            continue
        fileParts = d.split(os.path.sep)[-1].split('_')
        if fileParts[0] not in subjectFiles.keys():
            subjectFiles[fileParts[0]] = [[],[]]

        subjectFiles[fileParts[0]][0].append(int(fileParts[2]))
        subjectFiles[fileParts[0]][1].append(d)

    for s in subjectFiles:
        fileRightOrder = subjectFiles[s][0]
        fileRightOrder.sort()
        for f in subjectFiles[s][1]:
            brokenFile = f.split(os.path.sep)[-1].split('_')
            shutil.copyfile(f,os.path.join(args.rightDIFolder, brokenFile[0]+ '_%04d_' % (fileRightOrder.index(int(brokenFile[2]))+1 ) + '_'.join(brokenFile[2:]) ))
            for axis in ['x','y']:
                for dg in range(-30,40,10):
                    if dg == 0:
                        continue
                    fileNameRotate = brokenFile[0]+ '_%04d_%s_%s_rotate_%d_%s_newdepth_%s' % (fileRightOrder.index(int(brokenFile[2]) )+1,brokenFile[2],brokenFile[3],dg,axis,brokenFile[-1])
                    originalFileName = brokenFile[0]+ '_%s_%s_%s_rotate_%d_%s_newdepth_%s' % (brokenFile[1],brokenFile[2],brokenFile[3],dg,axis,brokenFile[-1])
                    shutil.copyfile(os.path.join(args.DIFolder,originalFileName),os.path.join(args.rightDIFolder, fileNameRotate))



if __name__ == '__main__':
    main()