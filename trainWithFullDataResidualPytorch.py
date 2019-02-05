from helper.functions import generateData, generateFoldsOfData, generateImageData, loadFoldFromFolders
import networks.PyTorch.jojo as jojo, argparse, numpy as np, torch, torch.optim as optim, torch.nn.functional as F
import torch.utils.data, shutil, os
from tensorboardX import SummaryWriter
from PyTorchLayers.center_loss import CenterLoss

def save_checkpoint(state,is_best,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-p', '--pathBase', default='generated_images_lbp_frgc', help='Path for faces', required=False)
    parser.add_argument('-b', '--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('-c', '--classNumber', type=int, default=466, help='Quantity of classes', required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to be run', required=False)
    parser.add_argument('--output', default=None, help='Output Folder', required=False)
    parser.add_argument('--residualFolder', default='overflowMasks_pr__underflowMasks_pr',help='Folder with the residual information', required=False)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Carregando dados')
    imageData, classesData = generateData(args.pathBase)
    folds = generateFoldsOfData(10, imageData, classesData)

    imageData = folds[0][0]
    classesData = folds[0][1]

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    print('Criando diretorio')
    os.makedirs(args.output)
    muda = jojo.Joseph(args.classNumber)
    muda.to(device)

    qntBatches = len(imageData) / args.batch

    '''
    if args.centerLoss:
        center_loss_l = CenterLoss(num_classes=args.classNumber)
        optimizer_celoss = optim.SGD(center_loss_l.parameters(), lr=0.01, momentum=0.5)
    '''
    foldProbe = generateImageData(folds[0][2], loadMasks=args.residualFolder.split('__'),prefix=False)
    foldProbeClasses = np.array(folds[0][3]) - 1

    foldProbe = torch.from_numpy(np.rollaxis(foldProbe, 3, 1)).float()
    foldProbeClasses = torch.from_numpy(foldProbeClasses)
    pdata = torch.utils.data.TensorDataset(foldProbe, foldProbeClasses)
    test_loader = torch.utils.data.DataLoader(pdata, batch_size=args.batch, shuffle=False)

    print('Criando otimizadores')
    optimizer = optim.SGD(muda.parameters(), lr=0.01, momentum=0.5)

    print('Iniciando treino')

    cc = SummaryWriter()
    bestForFold = -1

    for ep in range(args.epochs):
        lossAcc = []
        for i in range(0,len(imageData),3000):
            print('Criando tensores')
            dataBatch = np.array(
            generateImageData(imageData[i:i+3000], loadMasks=args.residualFolder.split('__'), prefix=False,silent=True)) / 255.0
            classesBatch = np.array(classesData[i:i+3000]) - 1

            dataBatch = torch.from_numpy(np.rollaxis(dataBatch, 3, 1)).float()
            classesBatch = torch.from_numpy(classesBatch)
            tdata = torch.utils.data.TensorDataset(dataBatch, classesBatch)
            train_loader = torch.utils.data.DataLoader(tdata, batch_size=args.batch, shuffle=True)

            for bIdx, (currBatch, currTargetBatch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = muda(currBatch.to(device))
                loss = F.cross_entropy(output, currTargetBatch.to(device))
                loss.backward()
                optimizer.step()
                lossAcc.append(loss.item())

                cc.add_scalar('GioGioFullTraining/fullData/loss', sum(lossAcc) / len(lossAcc), ep)

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

        cc.add_scalar('GioGioFullTraining/fullData/accuracy', cResult, ep)

        print('[EPOCH %d] Accuracy of the network on the %d test images: %d %% Loss %f' % (ep,total,100 * cResult,sum(lossAcc) / len(lossAcc)))

        if bestForFold < cResult:
            fName = '%s_best.pth.tar' % ('GioGio')
            fName = os.path.join(args.output, fName)
            save_checkpoint({
                'epoch': ep + 1,
                'arch': 'GioGio',
                'state_dict': muda.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False, fName)

            bestForFold = cResult

    print('Terminou')