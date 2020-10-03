from torchvision.transforms import transforms
from datasetClass.structures import loadFolder
from helper.functions import getFilesInPath
import argparse, networks.PyTorch.jojo as jojo
import torch.utils.data, os, numpy as np, shutil
from joblib import load

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Deep Features utilizing GioGio Network')
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=True)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=True)
    parser.add_argument('--output', default=None, help='Features output files', required=True)
    parser.add_argument('--network', help='Joestar network to use', required=False, default='giogio')
    parser.add_argument('--pca', help='PCA', required=False, default=False)
    parser.add_argument('--modeLoadFile', help='Mode to load', required=False, default='auto')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([176.95474987005545,176.95474987005545,176.95474987005545],[20.804202200767897,20.804202200767897,20.804202200767897])
    ])

    paths = getFilesInPath(args.loadFromFolder)
    foldFile = loadFolder(args.loadFromFolder, args.modeLoadFile, dataTransform)
    gal_loader = torch.utils.data.DataLoader(foldFile, batch_size=100, shuffle=False)

    checkpoint = torch.load(args.fineTuneWeights)
    if args.network == 'giogio':
        muda = jojo.GioGio(checkpoint['state_dict']['softmax.2.weight'].shape[0],in_channels=checkpoint['state_dict']['features.0.weight'].shape[1]).to(device)
    elif args.network == 'jolyne':
        channelsIn = 4 if args.modeLoadFile == 'RGBA' else 3
        muda = jojo.Jolyne(checkpoint['state_dict']['softmax.2.weight'].shape[0],
                           in_channels=channelsIn).to(device)
    muda.load_state_dict(checkpoint['state_dict'])

    muda.eval()
    galleryFeatures = []
    galleryClasses = []
    filePathNameIdx = 0
    with open(args.output, 'w') as dk:
        with torch.no_grad():
            for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
                print("Extracting features from batch %d"%(bIdx))
                output, whatever = muda(currBatch.to(device))
                galleryFeatures = whatever.tolist()
                galleryClasses = currTargetBatch.tolist()

                for i, data in enumerate(galleryFeatures):
                    filesName = paths[filePathNameIdx].split(os.path.sep)[-1]
                    filePathNameIdx += 1
                    dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName +'\n')


    if args.pca:
        pca = load(args.pca)
        galleryFeatures=np.array(galleryFeatures)
        galleryFeatures=pca.transform(galleryFeatures).tolist()
    '''
    print("Quantidade de características: %d" % (len(galleryFeatures[0])))
    print('Writing feature file')
    with open(args.output, 'w') as dk:
        for i, data in enumerate(galleryFeatures):
            filesName = paths[i].split(os.path.sep)[-1]
            dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName +'\n')
    '''