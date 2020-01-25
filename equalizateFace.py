import argparse, cv2, os, shutil, numpy as np
from helper.functions import getFilesInPath, scaleValues

def main():
    parser = argparse.ArgumentParser(description='Normalize depth values')
    parser.add_argument('--pathDatabase', help='Path for the database', required=True)
    parser.add_argument('--folderResult', help='Folder with output', required=True)
    args = parser.parse_args()


    if not os.path.exists(args.folderResult):
        os.makedirs(args.folderResult)

    filesInPath = getFilesInPath(args.pathDatabase)
    sumAverage = np.zeros((1,len(filesInPath))).flatten()
    for idx, f in enumerate(filesInPath):
        imageFile = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
        sumAverage[idx] = np.mean(imageFile)
        newImageFile = imageFile.copy()
        newImageFile[50:,0:] = 255
        fileName = f.split(os.path.sep)[-1]
        newfileName = fileName.split('.')
        newfileName[0] = newfileName[0][:-9]+'_occluded_lower_newdepth'
        cv2.imwrite(os.path.join(args.folderResult,'.'.join(newfileName)),newImageFile)

        newImageFile = imageFile.copy()
        newImageFile[0:,50:] = 255
        newfileName = fileName.split('.')
        newfileName[0] = newfileName[0][:-9]+'_occluded_right_newdepth'
        cv2.imwrite(os.path.join(args.folderResult,'.'.join(newfileName)),newImageFile)

        newImageFile = imageFile.copy()
        newImageFile[0:,0:50] = 255
        newfileName = fileName.split('.')
        newfileName[0] = newfileName[0][:-9]+'_occluded_left_newdepth'
        cv2.imwrite(os.path.join(args.folderResult,'.'.join(newfileName)),newImageFile)

    print(np.mean(sumAverage))
    print(np.std(sumAverage))

if __name__ == '__main__':
    main()