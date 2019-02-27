from helper.functions import generateData, generateFoldsOfData, generateImageData, loadFoldFromFolders
import networks.PyTorch.jojo as jojo, argparse, numpy as np, torch, torch.optim as optim, torch.nn.functional as F
import torch.utils.data, shutil, os, re
from networks.PyTorch.vgg_face_dag import *

features = []

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    if output.device.type == 'cuda':
        features.append(output.data.cpu().numpy())
    else:
        features.append(output.data.numpy())

def filterFiles(pathFiles, classes, regularExpression):
    returnFileData = []
    returnImageClasses = []
    for i, f in enumerate(pathFiles):
        fileName = f.split(os.path.sep)[-1]
        for rep in regularExpression:
            if re.match(rep,fileName):
                returnFileData.append(f)
                returnImageClasses.append(classes[i])
                break

    return returnFileData, returnImageClasses

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Deep Features utilizing GioGio Network')
    parser.add_argument('-p','--pathdatabase',help='Path for the database',required=True)
    parser.add_argument('-w', '--weights', help='File for weights', required=True)
    parser.add_argument('-o', '--output', help='Output for features', default='out.txt')
    parser.add_argument('-g', '--gallery', help='Gallery', default='gallery.txt')
    parser.add_argument('-r', '--probe', help='Probe', default='probe.txt')
    parser.add_argument('-f', '--folds', help='Quantity of folds', default=None, type=int)
    parser.add_argument('--exp', help='Regular expression', default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageData, classesData = generateData(args.pathdatabase)
    if args.exp is not None:
        imageData, classesData = filterFiles(imageData, classesData, args.exp.split('__'))
        
    imageData = generateImageData(imageData) / 255.0
    classesData = np.array(classesData)

    if np.amin(classesData) == 1:
        classesData = classesData - 1

    muda = vgg_face_dag()
    muda.conv3_1 = nn.Conv2d(4, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
    muda.fc6 = nn.Linear(in_features=256 * 12 * 12, out_features=4096, bias=True)
    muda = vgg_smaller(muda)
    muda.fullyConnected.add_module('new_softmax', nn.Linear(2622, 458))
    state_dict = torch.load(args.weights)
    muda.load_state_dict(state_dict['state_dict'])
    muda.to(device)

    muda.fullyConnected[-2].register_forward_hook(printnorm)

    foldGallery = torch.from_numpy(np.rollaxis(imageData, 3, 1)).float()
    foldGalleryClasses = torch.from_numpy(classesData)
    tdata = torch.utils.data.TensorDataset(foldGallery, foldGalleryClasses)
    train_loader = torch.utils.data.DataLoader(tdata, batch_size=100, shuffle=False)

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            images, labels = data
            pred = muda(images.to(device))

    with open(args.output,'w') as dk:
        classIdx = 0
        for i, data in enumerate(features):
            for d in data:
                dk.write(' '.join(list(map(str,d))) + ' ' + str(classesData[classIdx]) + '\n')
                classIdx += 1

    print(features)