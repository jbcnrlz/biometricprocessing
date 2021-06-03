from helper.functions import getFilesInPath
import argparse, os, shutil, cv2

def main():
    parser = argparse.ArgumentParser(description='Join depth folders')
    parser.add_argument('--depthFolderOriginal', help='Path for Original Folder files', nargs='+', required=True)
    parser.add_argument('--depthFolderJoined', help='Joined folder depth', required=True)
    parser.add_argument('--DIsFolder', help='DIs folder', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.depthFolderJoined):
        os.makedirs(args.depthFolderJoined)

    depthsFiles = [ getFilesInPath(dfs) for dfs in args.depthFolderOriginal ]

    dis = getFilesInPath(args.DIsFolder)
    for d in dis:
        dFileName = d.split(os.path.sep)[-1]
        dFileName = dFileName.split('.')[0]
        if os.path.exists(os.path.join(args.depthFolderJoined,dFileName+'.jpeg')):
            continue
        for dfs in depthsFiles:
            for df in dfs:
                fileName = df.split(os.path.sep)[-1]
                fileNameNoEXT = fileName.split('.')[0]
                if dFileName == fileNameNoEXT:
                    shutil.copy(df,os.path.join(args.depthFolderJoined,fileName))
                    break
                else:
                    oldFileNameSplt = dFileName.split('_')
                    if 'bs' in oldFileNameSplt[1] and 'bs' in fileNameNoEXT:
                        subjectClass = int(oldFileNameSplt[1][2:])
                        oldFileName = oldFileNameSplt.copy()
                        oldFileName[0] = '%03d' % subjectClass
                        oldFileName[4] = '0'
                        oldFileName = '_'.join(oldFileName)
                        if oldFileName == fileNameNoEXT:
                            newFileName = '_'.join(oldFileNameSplt)
                            fileDepth = cv2.imread(df)
                            cv2.imwrite(os.path.join(args.depthFolderJoined,newFileName+'.jpeg'),fileDepth)

                            #print(os.path.join(args.depthFolderJoined,newFileName+'.jpeg'))
                            #shutil.copy(df,os.path.join(args.depthFolderJoined,newFileName+'.jpeg'))
                            break


if __name__ == '__main__':
    main()