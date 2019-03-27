from torchvision import transforms
from helper.functions import plot_confusion_matrix, saveStatePytorch
from datasetClass.structures import loadFoldsDatasets, loadDatasetFromFolder
import networks.PyTorch.vgg_face_dag as vgg
import networks.PyTorch.jojo as jojo, argparse, numpy as np, torch, torch.optim as optim, torch.nn.functional as F
import torch.utils.data, shutil, os, time, torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn as nn
from PyTorchLayers.center_loss import CenterLoss

imagesForTensorboard = []

def shortenNetwork(network,desiredLayers):
    return [network[i] for i in desiredLayers]

def save_checkpoint(state,is_best,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def saveImageLayer(self, input, output):
    if output.device.type == 'cuda':
        layerValue = output.data.cpu()
    else:
        layerValue = output.data

    for i in range(layerValue.shape[0]):
        for j in range(layerValue.shape[1]):
            x = vutils.make_grid(layerValue[i][j], normalize=True, scale_each=True)
            imagesForTensorboard.append(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-p', '--pathBase',default='generated_images_lbp_frgc',help='Path for faces', required=False)
    parser.add_argument('-b', '--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('-c', '--classNumber', type=int, default=466, help='Quantity of classes', required=False)
    parser.add_argument('-t', '--runOnTest', type=bool, default=False, help='Run on test data', required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to be run', required=False)
    parser.add_argument('-f', '--folds', type=int, default=10, help='Fold quantity', required=False)
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=False)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=False)
    parser.add_argument('--useTensorboard', default=True, help='Utilize tensorboard', required=False, type=bool)
    parser.add_argument('--tensorBoardName', default='GioGioData', help='Teensorboard variable name', required=False)
    parser.add_argument('--startingFold', default=0, help='Teensorboard variable name', required=False, type=int)
    parser.add_argument('--fineTuningClasses', default=0, help='Fine Tuning classes number', required=False, type=int)
    parser.add_argument('--folderSnapshots', default='trainPytorch', help='Folder for snapshots', required=False)
    parser.add_argument('--extension', help='Extension from files', required=False, default='png')
    parser.add_argument('--arc', help='Network', required=False, default='giogio')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = None
    if args.arc.lower() == 'vgg':
        dataTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.562454871481894, 0.8208898956471341, 0.395364053852456],
                                 std=[0.43727472598867456, 0.31812502566122625, 0.3796120355707891])
        ])
    else:
        dataTransform = transforms.Compose([
            transforms.ToTensor()
        ])


    in_channels = 4 if args.extension == 'png' else 3

    if args.loadFromFolder is None:
        folds = [loadDatasetFromFolder(args.pathBase, validationSize='auto', transforms=dataTransform)]
        #imageData, classesData = generateData(args.pathBase,extension=args.extension)
        #folds = generateFoldsOfData(args.folds,imageData,classesData)

    else:
        folds = loadFoldsDatasets(args.loadFromFolder,dataTransform)
        #folds = loadFoldFromFolders(args.loadFromFolder)

    if os.path.exists(args.folderSnapshots):
        shutil.rmtree(args.folderSnapshots)
    os.makedirs(args.folderSnapshots)

    if args.useTensorboard:
        cc = SummaryWriter()

    folds = folds[args.startingFold:]

    foldResults = []
    trainTimeFolds = []
    for f, datas in enumerate(folds):
        bestForFold = -1
        bestLossForFold =10000000
        epochBestForFold = None
        print('====================== Fazendo Fold =======================')
        if args.arc.lower() == 'vgg':
            muda = vgg.vgg_face_dag_load(weights_path=args.fineTuneWeights)
            muda = vgg.vgg_smaller(muda)
            shortenedList = shortenNetwork(
                list(muda.convolutional.children()),
                [0, 1, 4, 5, 6, 9, 10, 11, 16]
            )
            muda.convolutional = nn.Sequential(*shortenedList)
            muda.fullyConnected.add_module('7', nn.Linear(in_features=2622, out_features=args.fineTuningClasses))
        else:
            muda = jojo.GioGio(args.classNumber,in_channels=in_channels)
            if args.fineTuneWeights is not None:
                checkpoint = torch.load(args.fineTuneWeights)
                #optimizer.load_state_dict(checkpoint['optimizer'])
                muda.load_state_dict(checkpoint['state_dict'])
                nfeats = muda.softmax[-1].in_features
                muda.softmax[-1] = nn.Linear(nfeats, args.fineTuningClasses)

        print(muda)
        muda.to(device)

        foldResults.append([])
        train_loader = torch.utils.data.DataLoader(datas[0], batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datas[1], batch_size=args.batch, shuffle=False)

        optimizer = optim.SGD(muda.parameters(), lr=0.01, momentum=0.5)
        closs = CenterLoss(args.fineTuningClasses, 4096).to(device)
        optim_closs = optim.SGD(closs.parameters(), lr=0.6)
        criterion = nn.CrossEntropyLoss().to(device)
        alpha = 0.003

        if not os.path.exists(os.path.join(args.folderSnapshots, str(f))):
            os.makedirs(os.path.join(args.folderSnapshots, str(f)))

        bestResult = -1
        bestEpoch = -1

        fTrainignTime = []
        #muda.features[-1].register_forward_hook(saveImageLayer)
        for ep in range(args.epochs):
            ibl = ibr = ' '
            muda.train()
            lossAcc = []
            start_time = time.time()
            for bIdx, (currBatch,currTargetBatch) in enumerate(train_loader):
                currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)

                output, features = muda(currBatch)

                loss = criterion(output, currTargetBatch)
                #loss = loss + (alpha * closs(currTargetBatch, features))

                optimizer.zero_grad()
                #optim_closs.zero_grad()
                loss.backward()
                optimizer.step()
                #optim_closs.step()

                lossAcc.append(loss.item())

            finishTime = time.time() - start_time
            fTrainignTime.append(finishTime)

            if args.useTensorboard:
                cc.add_scalar(args.tensorBoardName+'/fold_'+str(args.startingFold+f+1)+'/loss', sum(lossAcc) / len(lossAcc), ep)

            muda.eval()
            total = 0
            correct = 0
            labelsData = [[],[]]
            scores = []
            loss_val = []
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    outputs, fs = muda(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels.to(device))
                    loss_val.append(loss)

                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
                    labelsData[0] = labelsData[0] + np.array(labels.cpu()).tolist()
                    labelsData[1] = labelsData[1] + np.array(predicted.cpu()).tolist()

            cResult = correct / total
            lossAvg = sum(lossAcc) / len(lossAcc)
            tLoss = sum(loss_val) / len(loss_val)

            if args.useTensorboard:
                cc.add_scalar(args.tensorBoardName+'/fold_'+str(args.startingFold+f+1)+'/Validation_loss', tLoss, ep)

            state_dict = muda.state_dict()
            opt_dict = optimizer.state_dict()

            if bestForFold < cResult:
                ibr = 'X'
                fName = '%s_best_rank.pth.tar' % ('GioGio')
                fName = os.path.join(args.folderSnapshots, str(f),fName)
                saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
                bestForFold = cResult

            if bestLossForFold > lossAvg:
                ibl = 'X'
                fName = '%s_best_loss.pth.tar' % ('GioGio')
                fName = os.path.join(args.folderSnapshots, str(f),fName)
                saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
                bestLossForFold = lossAvg

            if args.useTensorboard:
                cc.add_scalar(args.tensorBoardName+'/fold_' + str(args.startingFold+f+1) + '/accuracy', cResult, ep)

            if bestResult < cResult:
                bestResult = cResult
                bestEpoch = ep

            if ep % 10 == 0:
                fName = '%s_checkpoint_%05d.pth.tar' % ('GioGio',ep)
                fName = os.path.join(args.folderSnapshots, str(f),fName)
                save_checkpoint({
                    'epoch': ep + 1,
                    'arch': 'GioGio',
                    'state_dict': muda.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, False, fName)
                foldResults[-1].append(correct / total)
                a = [i for i in range(max(labelsData[0])+1)]
                confMat = plot_confusion_matrix(labelsData[0],labelsData[1],['Subject '+str(lnm) for lnm in a])
                cc.add_figure(args.tensorBoardName+'/fold_' + str(args.startingFold+f+1) + '/confMatrix', confMat,ep)


            print('[EPOCH %03d] Accuracy of the network on the %d validating images: %03.2f %% Training Loss %.5f Validation Loss %.5f [%c] [%c]' % (ep, total, 100 * cResult, lossAvg, tLoss, ibl, ibr))
        trainTimeFolds.append(sum(fTrainignTime))
        print('Best result %2.6f Epoch %d' % (bestResult*100,bestEpoch))