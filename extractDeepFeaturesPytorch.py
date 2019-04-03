from torchvision.transforms import transforms
from datasetClass.structures import loadFolder
from helper.functions import getFilesInPath
import argparse, networks.PyTorch.jojo as jojo
import torch.utils.data, os, numpy as np
from sklearn.decomposition import PCA

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Deep Features utilizing GioGio Network')
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=True)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=True)
    parser.add_argument('--output', default=None, help='Features output files', required=True)
    parser.add_argument('--network', help='Joestar network to use', required=False, default='giogio')
    parser.add_argument('--pca', help='PCA', required=False, default=False, type=bool)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    paths = getFilesInPath(args.loadFromFolder)
    foldFile = loadFolder(args.loadFromFolder, dataTransform)
    gal_loader = torch.utils.data.DataLoader(foldFile, batch_size=100, shuffle=False)

    checkpoint = torch.load(args.fineTuneWeights)
    if args.network == 'giogio':
        muda = jojo.GioGio(checkpoint['state_dict']['softmax.1.weight'].shape[0],in_channels=checkpoint['state_dict']['features.0.weight'].shape[1]).to(device)
    elif args.network == 'jolyne':
        muda = jojo.Jolyne(checkpoint['state_dict']['softmax.2.weight'].shape[0],
                           in_channels=checkpoint['state_dict']['features.0.weight'].shape[1]).to(device)
    muda.load_state_dict(checkpoint['state_dict'])

    muda.eval()
    galleryFeatures = []
    galleryClasses = []
    with torch.no_grad():
        for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
            print("Extracting features from batch %d"%(bIdx))
            output, whatever = muda(currBatch.to(device))
            galleryFeatures = galleryFeatures + whatever.tolist()
            galleryClasses = galleryClasses + currTargetBatch.tolist()

    if args.pca:
        pca = PCA(n_components=0.99,svd_solver='full')
        galleryFeatures=np.array(galleryFeatures)
        pca.fit(galleryFeatures)
        galleryFeatures=pca.transform(galleryFeatures).tolist()
    print("Quantidade de características: %d" % (len(galleryFeatures[0])))
    print('Writing feature file')
    with open(args.output, 'w') as dk:
        for i, data in enumerate(galleryFeatures):
            filesName = paths[i].split(os.path.sep)[-1]
            dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName +'\n')
