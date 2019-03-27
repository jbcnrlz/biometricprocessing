from torchvision.transforms import transforms
from datasetClass.structures import loadFolder, loadDatasetFromFolder
from helper.functions import getFilesInPath
from networks.PyTorch.jojo import GioGio
from networks.PyTorch.potara import Earings
import networks.PyTorch.vgg_face_dag as vgg, argparse, os, torch, shutil, torch.optim as optim
import torch.nn as nn, numpy as np
from helper.functions import saveStatePytorch, shortenNetwork
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuse two networks')
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=True)
    parser.add_argument('--vggWeights',    help='VGG Weights', required=False)
    parser.add_argument('--giogioWeights', help='GioGio Weights', required=False)
    parser.add_argument('--fusedweights', help='Fused Weights', required=False)
    parser.add_argument('--output', default=None, help='Features output files', required=True)
    parser.add_argument('--mode', default='feature', help='Mode of network', required=False)
    parser.add_argument('--epochs', help='Epochs to pretrain', required=False, type=int)
    parser.add_argument('-b', '--batch', type=int, default=20, help='Size of the batch', required=False)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.mode == 'feature':

        print('Criando Modelos')
        vggPart = vgg.vgg_face_dag_load()
        vggPart = vgg.vgg_smaller(vggPart, 458)
        shortenedList = shortenNetwork(
            list(vggPart.convolutional.children()),
            [12, 13, 16, 17, 18, 23], True
        )
        vggPart.convolutional = nn.Sequential(*shortenedList)
        vggPart.fullyConnected = vggPart.fullyConnected[1:]
        vggPart.to(device)

        muda = GioGio(458, in_channels=3).to(device)
        muda.features = muda.features[:-5]
        giov = Earings(muda,vggPart).to(device)
        checkpoint = torch.load(args.fusedweights)
        giov.load_state_dict(checkpoint['state_dict'])
        print(giov)

        paths = getFilesInPath(args.loadFromFolder)
        foldFile = loadFolder(args.loadFromFolder, dataTransform)
        gal_loader = torch.utils.data.DataLoader(foldFile, batch_size=15, shuffle=False)
        galleryFeatures = []
        galleryClasses = []
        giov.eval()
        with torch.no_grad():
            for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
                output, whatever = giov(currBatch.to(device))
                galleryFeatures = galleryFeatures + output.tolist()
                galleryClasses = galleryClasses + currTargetBatch.tolist()

        print('Writing feature file')
        with open(args.output, 'w') as dk:
            for i, data in enumerate(galleryFeatures):
                filesName = paths[i].split(os.path.sep)[-1]
                dk.write(' '.join(list(map(str, data))) + ' ' + str(galleryClasses[i]) + ' ' + filesName +'\n')

    else:

        folds = loadDatasetFromFolder(args.loadFromFolder, validationSize='auto', transforms=dataTransform)

        gal_loader = torch.utils.data.DataLoader(folds[0], batch_size=args.batch, shuffle=True)
        pro_loader = torch.utils.data.DataLoader(folds[1], batch_size=args.batch, shuffle=False)

        if os.path.exists(args.output):
            shutil.rmtree(args.output)
        os.makedirs(args.output)

        print('Criando Modelos')
        vggPart = vgg.vgg_face_dag_load(weights_path=args.vggWeights)
        vggPart = vgg.vgg_smaller(vggPart, 458)
        print(vggPart)
        shortenedList = shortenNetwork(
            list(vggPart.convolutional.children()),
            [12, 13, 16, 17, 18, 23], True
        )
        vggPart.convolutional = nn.Sequential(*shortenedList)
        vggPart.fullyConnected = vggPart.fullyConnected[1:]
        vggPart.to(device)

        muda = GioGio(458, in_channels=3).to(device)
        muda.features = muda.features[:-5]
        giov = Earings(muda,vggPart).to(device)
        print(giov)

        print('Criando otimizadores')

        optimizer = optim.SGD(giov.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.8)
        criterion = nn.CrossEntropyLoss().to(device)

        print('Iniciando treino')

        cc = SummaryWriter()
        bestForFold = 500000
        bestRankForFold = -1
        ep = 0
        while True:
            giov.train()
            scheduler.step()
            lossAcc = []

            for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
                currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)
                features, output = giov(currBatch)
                loss = criterion(output, currTargetBatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lossAcc.append(loss.item())

            lossAvg = sum(lossAcc) / len(lossAcc)
            cc.add_scalar('GIOV/loss', lossAvg, ep)

            giov.eval()
            total = 0
            correct = 0
            labelsData = [[], []]
            lossV = []
            probeFeatures = []
            probeClasses = []

            with torch.no_grad():
                for data in pro_loader:
                    images, labels = data
                    fs, outputs = giov(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
                    labelsData[0] = labelsData[0] + np.array(labels).tolist()
                    labelsData[1] = labelsData[1] + np.array(predicted).tolist()

                    probeFeatures.append(fs.data.cpu().numpy())
                    probeClasses.append(labels.data.cpu().numpy())

            cResult = correct / total

            cc.add_scalar('GIOV/accuracy', cResult, ep)

            print('[EPOCH %d] Accuracy of the network on the %d test images: %.2f%% Loss %f' % (
            ep, total, 100 * cResult, lossAvg))

            print('Salvando epoch atual')
            state_dict = giov.state_dict()
            opt_dict = optimizer.state_dict()
            fName = '%s_current.pth.tar' % ('giov')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)

            if bestRankForFold < cResult:
                print('Salvando melhor rank')
                fName = '%s_best_rank.pth.tar' % ('giov')
                fName = os.path.join(args.output, fName)
                saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
                bestRankForFold = cResult

            if bestForFold > lossAvg:
                print('Salvando melhor Loss')
                fName = '%s_best_loss.pth.tar' % ('giov')
                fName = os.path.join(args.output, fName)
                saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
                bestForFold = lossAvg

            ep += 1
            if args.epochs < 0:
                if lossAvg < 0.0001:
                    break
            else:
                args.epochs -= 1
                if args.epochs < 0:
                    break

        print('Terminou')