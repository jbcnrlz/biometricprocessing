from helper.functions import generateData, generateFoldsOfData, generateImageData, loadFoldFromFolders, scaleValues, plot_confusion_matrix
import networks.PyTorch.jojo as jojo, argparse, numpy as np, torch, torch.optim as optim, torch.nn.functional as F
import torch.utils.data, shutil, os, time, torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn as nn
from sklearn.metrics import confusion_matrix

imagesForTensorboard = []

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
    parser.add_argument('--residualFolder', default='overflowMasks_pr__underflowMasks_pr', help='Folder with the residual information', required=False)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.loadFromFolder is None:
        imageData, classesData = generateData(args.pathBase)
        folds = generateFoldsOfData(args.folds,imageData,classesData)

    else:
        folds = loadFoldFromFolders(args.loadFromFolder)

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
        epochBestForFold = None
        print('====================== Fazendo Fold =======================')
        muda = jojo.Joseph(args.classNumber)
        muda.to(device)

        foldResults.append([])
        foldProbe = generateImageData(datas[2],loadMasks=args.residualFolder.split('__'))
        foldProbeClasses = np.array(datas[3])
        foldGallery = generateImageData(datas[0],loadMasks=args.residualFolder.split('__'))
        foldGalleryClasses = np.array(datas[1])

        foldGallery = foldGallery / 255
        foldProbe = foldProbe / 255
        if np.amin(foldGalleryClasses) == 1:
            foldGalleryClasses = foldGalleryClasses - 1
            foldProbeClasses = foldProbeClasses - 1

        qntBatches = foldGallery.shape[0] / args.batch
        foldGallery = torch.from_numpy(np.rollaxis(foldGallery, 3, 1)).float()
        foldGalleryClasses = torch.from_numpy(foldGalleryClasses).to(device)
        tdata = torch.utils.data.TensorDataset(foldGallery, foldGalleryClasses)
        train_loader = torch.utils.data.DataLoader(tdata, batch_size=args.batch, shuffle=True)

        foldProbe = torch.from_numpy(np.rollaxis(foldProbe, 3, 1)).float()
        foldProbeClasses = torch.from_numpy(foldProbeClasses)
        pdata = torch.utils.data.TensorDataset(foldProbe, foldProbeClasses)
        test_loader = torch.utils.data.DataLoader(pdata, batch_size=args.batch, shuffle=False)

        optimizer = optim.SGD(muda.parameters(), lr=0.01, momentum=0.5)

        if not os.path.exists(os.path.join(args.folderSnapshots, str(f))):
            os.makedirs(os.path.join(args.folderSnapshots, str(f)))

        if args.fineTuneWeights is not None:
            checkpoint = torch.load(args.fineTuneWeights)
            optimizer.load_state_dict(checkpoint['optimizer'])
            muda.load_state_dict(checkpoint['state_dict'])
            nfeats = muda.classifier[-1].in_features
            muda.add_module('new_softmax',nn.Linear(nfeats, args.fineTuningClasses).to(device))

        print(muda)

        bestResult = -1
        bestEpoch = -1

        fTrainignTime = []
        for ep in range(args.epochs):
            muda.train()
            lossAcc = []
            start_time = time.time()
            for bIdx, (currBatch,currTargetBatch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = muda(currBatch.to(device))
                loss = F.cross_entropy(output, currTargetBatch.to(device))
                loss.backward()
                optimizer.step()
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
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    outputs = muda(images.to(device))
                    scores = scores + np.array(outputs.data).tolist()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device) ).sum().item()
                    labelsData[0] = labelsData[0] + np.array(labels).tolist()
                    labelsData[1] = labelsData[1] + np.array(predicted).tolist()

            cResult = correct / total

            print('[EPOCH %d] Accuracy of the network on the %d test images: %.2f %% Loss %f' % (ep,total,100 * cResult,sum(lossAcc) / len(lossAcc)))

            if bestForFold < cResult:
                fName = '%s_best.pth.tar' % ('GioGio')
                fName = os.path.join(args.folderSnapshots, str(f),fName)
                save_checkpoint({
                    'epoch': ep + 1,
                    'arch': 'GioGio',
                    'state_dict': muda.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, False, fName)

                bestForFold = cResult
                with open('scores_fold_' + str(f) + '.txt', 'w') as ofs:
                    for ddv in scores:
                        ofs.write(' '.join(list(map(str,ddv))) + '\n')

                with open('labels_fold_' + str(f) + '.txt', 'w') as ofs:
                    for ddv in labelsData[0]:
                        ofs.write(str(ddv) + '\n')


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


        trainTimeFolds.append(sum(fTrainignTime))
        print('Best result %2.6f Epoch %d' % (bestResult*100,bestEpoch))


    print(foldResults)
    print(trainTimeFolds)