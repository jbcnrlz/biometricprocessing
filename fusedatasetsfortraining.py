from helper.functions import getFilesInPath
import shutil, argparse, os

def main():
    parser = argparse.ArgumentParser(description='Fuse datasets')
    parser.add_argument('--fusedPath',help='Path for new folder', required=True)
    parser.add_argument('--foldersDataset', nargs='+', help='Folders to join dataset', required=True)
    args = parser.parse_args()

    if os.path.exists(args.fusedPath):
        shutil.rmtree(args.fusedPath)

    os.makedirs(args.fusedPath)

    finalSubject = 0
    for idx, f in enumerate(args.foldersDataset):
        filesInDir = getFilesInPath(f)
        sorted_files = sorted(filesInDir, key=lambda x: int(x.split(os.path.sep)[-1].split('_')[0]) )
        finalSubject += int(idx > 0 and int(sorted_files[0].split(os.path.sep)[-1].split('_')[0]) == 0 )
        for fid in sorted_files:
            fileName = fid.split(os.path.sep)[-1]
            subjectNumber = int(fileName.split('_')[0])
            if idx == 0:
                shutil.copyfile(fid,os.path.join(args.fusedPath,fileName))
            else:
                nFileName = fileName.split('_')
                nSubbectNumber = str(finalSubject+subjectNumber)
                nFileName[0] = nSubbectNumber
                nFileName[4] = nSubbectNumber
                shutil.copyfile(fid, os.path.join(args.fusedPath, '_'.join(nFileName)))

        finalSubject = int(sorted_files[-1].split(os.path.sep)[-1].split('_')[0])

if __name__ == '__main__':
    main()