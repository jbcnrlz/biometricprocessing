from helper.functions import getFilesInPath
import cv2, os

def writeFeatureFile(pathFile,data):
    with open(pathFile,'a') as pf:
        for d in data:
            pf.write(' '.join(d) + '\n')

def main():
    filesDir = getFilesInPath('generated_images_eurecom')
    totalFeature = []
    for idx, f in enumerate(filesDir):
        if 'rotate' in f:
            continue
        print("Fazendo %d de %d" % (idx, len(filesDir)))
        image = cv2.imread(f,cv2.IMREAD_UNCHANGED)
        filename = f.split(os.path.sep)[-1]
        className = filename.split('_')[0]
        featureFace = list(map(str,image.flatten())) + [className] + [filename]
        #totalFeature.append(featureFace)

        writeFeatureFile('features/features_di_3dlbp.txt',[featureFace])

if __name__ == '__main__':
    main()