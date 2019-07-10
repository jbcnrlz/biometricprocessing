from torchvision.transforms import transforms
from datasetClass.structures import loadFolder, loadSiameseDatasetFromFolder
from helper.functions import getFilesInPath
import argparse, networks.PyTorch.jojo as jojo
import torch.utils.data, os, numpy as np
from joblib import load

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Deep Features utilizing Shallow Network - Siamese Like approach')
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=True)
    parser.add_argument('--others', help='Path other siamese data', required=True)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=True)
    parser.add_argument('--output', default=None, help='Features output files', required=True)
    parser.add_argument('--pca', help='PCA', required=False, default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    folds = loadSiameseDatasetFromFolder(args.loadFromFolder, otherFolds=args.others.split('__'), validationSize=None,
                                         transforms=dataTransform)
    gal_loader = torch.utils.data.DataLoader(folds, batch_size=100, shuffle=False)

    checkpoint = torch.load(args.fineTuneWeights)
    muda = jojo.SyameseJolyne(checkpoint['state_dict']['softmax.2.weight'].shape[0]).to(device)
    muda.load_state_dict(checkpoint['state_dict'])

    muda.eval()
    galleryFeatures = []
    galleryClasses = []
    with torch.no_grad():
        for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
            print("Extracting features from batch %d"%(bIdx))
            currBatch[0] = currBatch[0].to(device)
            currBatch[1] = currBatch[1].to(device)
            output, whatever = muda(currBatch)
            galleryFeatures = galleryFeatures + whatever.tolist()
            galleryClasses = galleryClasses + currTargetBatch.tolist()

    if args.pca:
        pca = load(args.pca)
        galleryFeatures=np.array(galleryFeatures)
        galleryFeatures=pca.transform(galleryFeatures).tolist()

    paths = getFilesInPath(args.loadFromFolder)
    print("Quantidade de características: %d" % (len(galleryFeatures[0])))
    print('Writing feature file')
    with open(args.output, 'w') as dk:
        for i, data in enumerate(galleryFeatures):
            filesName = paths[i].split(os.path.sep)[-1]
            dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName +'\n')
