from helper.functions import getFilesInPath
import argparse, cv2, numpy as np

def main():
    parser = argparse.ArgumentParser(description='Extract average from layers')
    parser.add_argument('--pathBase',default='generated_images_lbp_frgc',help='Path for faces', required=False)
    args = parser.parse_args()

    files = getFilesInPath(args.pathBase)
    chans = [[],[],[],[]]
    for f in files:
        if 'rotate' in f:
            continue
        imOpen = cv2.imread(f,cv2.IMREAD_UNCHANGED) / 255
        for i in range(imOpen.shape[2]):
            chans[i].append(imOpen[:,:,i])

    for imC in chans:
        print(np.mean(imC))
        print(np.std(imC))


if __name__ == '__main__':
    main()