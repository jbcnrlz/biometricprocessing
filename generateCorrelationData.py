import argparse, os, cv2, numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Generate Correlation Data')
    parser.add_argument('--sigmoidFile', help='Path for files to separate channels', required=True)
    parser.add_argument('--tdlbpFile', help='Path for files to separate channels', required=True)
    parser.add_argument('--rgbFile', help='Path for files to separate channels', required=True)
    parser.add_argument('--output', help='Folder to output to', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    sigmoidFile = cv2.imread(args.sigmoidFile,cv2.IMREAD_UNCHANGED)
    tdlbpFile = cv2.imread(args.tdlbpFile,cv2.IMREAD_UNCHANGED)
    rgbFile = cv2.imread(args.rgbFile,cv2.IMREAD_UNCHANGED)

    fig, axs = plt.subplots(3,3)
    fig.suptitle("Correlation between red and other channels")
    redLayer = 2
    compLayers = [0,1,3]
    channelName = ["Green","Blue","Red","Alpha"]
    titles = ["Sigmoid DI","3DLBP DI","RGB Image"]
    for idxIM, imgType in enumerate([sigmoidFile, tdlbpFile, rgbFile]):
        axs[0,idxIM].set_title(titles[idxIM])
        for idxCHAN,c in enumerate(compLayers):
            if c < imgType.shape[-1]:
                axs[idxCHAN,idxIM].scatter(imgType[:,:,redLayer].flatten(),imgType[:,:,c].flatten())
                axs[idxCHAN,idxIM].set(ylabel="Red VS "+channelName[c])

    for ax in axs.flat:
        ax.label_outer()

    plt.show()


if __name__ == '__main__':
    main()