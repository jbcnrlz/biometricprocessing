from torchvision.transforms import transforms
from datasetClass.structures import loadFolder
from helper.functions import getFilesInPath
import argparse, networks.PyTorch.jojo as jojo
import torch.utils.data, os, numpy as np, shutil
from joblib import load
from finetuneRESNET import initialize_model
import torch.nn as nn

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Deep Features utilizing GioGio Network')
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=True)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=True)
    parser.add_argument('--output', default=None, help='Features output files', required=True)
    parser.add_argument('--model', default='resnet50', help='Model to be loaded', required=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor(),
    ])

    paths = getFilesInPath(args.loadFromFolder)
    foldFile = loadFolder(args.loadFromFolder, 'RGBA', dataTransform)
    gal_loader = torch.utils.data.DataLoader(foldFile, batch_size=100, shuffle=False)

    checkpoint = torch.load(args.fineTuneWeights)

    muda, input_size = initialize_model(checkpoint['state_dict'][list(checkpoint['state_dict'].keys())[-2]].shape[0],True,False,args.model)
    muda.load_state_dict(checkpoint['state_dict'])
    muda.to(device)
    muda.fc = muda.fc[0:2]
    muda.eval()
    print(muda)
    galleryFeatures = []
    galleryClasses = []
    filePathNameIdx = 0
    with open(args.output, 'w') as dk:
        with torch.no_grad():
            for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
                print("Extracting features from batch %d"%(bIdx))
                output = muda(currBatch.to(device))
                galleryFeatures = output.reshape((-1,2048)).tolist()
                galleryClasses = currTargetBatch.tolist()

                for i, data in enumerate(galleryFeatures):
                    filesName = paths[filePathNameIdx].split(os.path.sep)[-1]
                    filePathNameIdx += 1
                    dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName +'\n')