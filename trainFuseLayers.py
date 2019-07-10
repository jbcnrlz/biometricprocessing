import networks.PyTorch.jojo as jojo, argparse, torch.optim as optim
import torch.utils.data, shutil, os
from helper.functions import saveStatePytorch
from tensorboardX import SummaryWriter
from datasetClass.structures import loadFeaturesFromText
from torchvision import transforms
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description='Train Fuse layers')
    parser.add_argument('--pathBase', help='Path for feature files (separated by __)', required=True)
    parser.add_argument('-b', '--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('-c', '--classNumber', type=int, default=466, help='Quantity of classes', required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to be run', required=False)
    parser.add_argument('--output', default=None, help='Output Folder', required=False)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.001, type=float)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    print('Carregando dados')
    folds = loadFeaturesFromText(args.pathBase, validationSize='auto', transforms=dataTransform)
    gal_loader = torch.utils.data.DataLoader(folds[0], batch_size=args.batch, shuffle=True)
    pro_loader = torch.utils.data.DataLoader(folds[1], batch_size=args.batch, shuffle=False)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    print('Criando diretorio')
    os.makedirs(args.output)
    muda = jojo.FusingNetwork(2048,args.classNumber)
    muda.to(device)

    print('Criando otimizadores')
    optimizer = optim.SGD(muda.parameters(),lr=args.learningRate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.8)
    criterion = nn.CrossEntropyLoss().to(device)

    print('Iniciando treino')

    cc = SummaryWriter()
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    print(muda)
    for ep in range(args.epochs):
        ibl = ibr = ibtl = ' '
        muda.train()
        scheduler.step()
        lossAcc = []
        for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
            #currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)
            currBatch[0] = currBatch[0].to(device)
            currBatch[1] = currBatch[1].to(device)
            output, features = muda(currBatch)

            loss = criterion(output, currTargetBatch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())

        lossAvg = sum(lossAcc) / len(lossAcc)
        cc.add_scalar('GioGioFullTraining/fullData/loss', lossAvg, ep)

        muda.eval()
        total = 0
        correct = 0
        loss_val = []
        with torch.no_grad():
            for bIdx, (currBatchProbe, currTargetBatchProbe) in enumerate(pro_loader):
                currBatchProbe[0] = currBatchProbe[0].to(device)
                currBatchProbe[1] = currBatchProbe[1].to(device)
                currTargetBatchProbe = currTargetBatchProbe.to(device)

                outputs, fs = muda(currBatchProbe)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, currTargetBatchProbe)
                loss_val.append(loss)

                total += currTargetBatchProbe.size(0)
                correct += (predicted == currTargetBatchProbe).sum().item()

        cResult = correct / total
        tLoss = sum(loss_val) / len(loss_val)
        cc.add_scalar('GioGioFullTraining/fullData/accuracy', cResult, ep)
        cc.add_scalar('GioGioFullTraining/fullData/Validation_loss', tLoss, ep)

        state_dict = muda.state_dict()
        opt_dict = optimizer.state_dict()
        fName = '%s_current.pth.tar' % ('jolyne')
        fName = os.path.join(args.output, fName)
        saveStatePytorch(fName, state_dict, opt_dict, ep + 1)

        if bestRankForFold < cResult:
            ibr = 'X'
            fName = '%s_best_rank.pth.tar' % ('jolyne')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestRankForFold = cResult

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % ('jolyne')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        if bestForFoldTLoss > tLoss:
            ibtl = 'X'
            fName = '%s_best_val_loss.pth.tar' % ('jolyne')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = tLoss

        print(
            '[EPOCH %03d] Accuracy of the network on the %d validating images: %.2f %% Training Loss %.5f Validation Loss %.5f [%c] [%c] [%c]' % (
            ep, total, 100 * cResult, lossAvg, tLoss, ibl, ibtl, ibr))

if __name__ == '__main__':
    main()