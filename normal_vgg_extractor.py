from torchvision.transforms import transforms
from datasetClass.structures import loadFolder
from helper.functions import getFilesInPath
import argparse, networks.PyTorch.vgg_face_dag as vgg
import torch
import torch.utils.data, os

def doFeatureExtration(model,loader,currFile,device,filePaths):
    galleryFeatures = []
    galleryClasses = []
    model.eval()
    with torch.no_grad():
        for bIdx, (currBatch, currTargetBatch) in enumerate(loader):
            print("Extracting features from batch %d"%(bIdx))
            output, whatever = model(currBatch.to(device))
            galleryFeatures = galleryFeatures + output.tolist()
            galleryClasses = galleryClasses + currTargetBatch.tolist()

    print('Writing feature file')
    with open(currFile, 'w') as dk:
        for i, data in enumerate(galleryFeatures):
            filesName = filePaths[i].split(os.path.sep)[-1]
            dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName +'\n')

def isFoldDir(pathFolder):
    return (os.path.exists(os.path.join(pathFolder,'1')) or os.path.exists(os.path.join(pathFolder,'0')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=True)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=True)
    parser.add_argument('--output', default=None, help='Features output files', required=True)
    parser.add_argument('--type', choices=['full','medium','small','smaller','finetuned','centerloss'], help='Type of vgg', required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.type == 'full':
        muda = vgg.vgg_face_dag_load(weights_path=args.fineTuneWeights).to(device)
        muda.fc6 = torch.nn.Linear(in_features=4608, out_features=4096, bias=True).to(device)
    elif args.type == 'medium':
        muda = vgg.medium_vgg_face_dag_load(weights_path=args.fineTuneWeights).to(device)
    elif args.type == 'finetuned':
        muda = vgg.finetuned_vgg_face_dag_load(weights_path=args.fineTuneWeights).to(device)
    elif args.type == 'small':
        muda = vgg.small_vgg_face_dag_load(weights_path=args.fineTuneWeights).to(device)
    elif args.type == 'smaller':
        muda = vgg.smaller_vgg_face_dag_load(weights_path=args.fineTuneWeights).to(device)
    elif args.type == 'centerloss':
        muda = vgg.centerloss_vgg_face_dag_load(weights_path=args.fineTuneWeights).to(device)

    dataTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.562454871481894, 0.8208898956471341, 0.395364053852456],
                             std=[0.43727472598867456, 0.31812502566122625, 0.3796120355707891])
    ])

    print(muda)
    print('Confirma extracao?')

    conf = input()

    if conf == 's':
        paths = getFilesInPath(args.loadFromFolder)
        foldFile = loadFolder(args.loadFromFolder,dataTransform)
        gal_loader = torch.utils.data.DataLoader(foldFile, batch_size=15, shuffle=False)
        doFeatureExtration(muda,gal_loader, args.output, device,paths)
