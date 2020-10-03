import networks.PyTorch.jojo as jojo, argparse, torch.optim as optim
import torch.utils.data, shutil, os
from helper.functions import saveStatePytorch
from tensorboardX import SummaryWriter
from datasetClass.structures import loadDatasetFromFolder, loadFoldsDatasets
from torchvision import transforms
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-p', '--pathBase', default='generated_images_lbp_frgc', help='Path for faces', required=False)
    parser.add_argument('-b', '--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('-c', '--classNumber', type=int, default=466, help='Quantity of classes', required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to be run', required=False)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=False)
    parser.add_argument('--output', default=None, help='Output Folder', required=False)
    parser.add_argument('--extension', help='Extension from files', required=False, default='png')
    parser.add_argument('--network', help='Joestar network to use', required=False, default='giogio')
    parser.add_argument('--layers', help='Quantitye of layers', required=False, default=None)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--tensorboardname', help='Learning Rate', required=False, default='GioGioFullTraining')
    parser.add_argument('--fineTuningClasses', default=0, help='Fine Tuning classes number', required=False, type=int)
    parser.add_argument('--meanImage', help='Mean image', nargs='+', required=False, type=float)
    parser.add_argument('--stdImage', help='Std image', nargs='+', required=False, type=float)
    parser.add_argument('--optimizer', help='Optimizer', required=False, default="sgd")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    print('Carregando dados')
    if os.path.exists(os.path.join(args.pathBase,'1')):
        folds = loadFoldsDatasets(args.pathBase, dataTransform)[0]
    else:
        folds = loadDatasetFromFolder(args.pathBase, validationSize='auto', transforms=dataTransform)
    gal_loader = torch.utils.data.DataLoader(folds[0], batch_size=args.batch, shuffle=True)
    pro_loader = torch.utils.data.DataLoader(folds[1], batch_size=args.batch, shuffle=False)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    print('Criando diretorio')
    if args.layers is None:
        in_channels = 4 if args.extension == 'png' else 3
    else:
        in_channels = int(args.layers)

    os.makedirs(args.output)
    if args.network == 'giogio':
        muda = jojo.GioGio(args.classNumber,in_channels=in_channels)
    elif args.network == 'jolyne':
        muda = jojo.Jolyne(args.classNumber, in_channels=in_channels)
    elif args.network == 'octjolyne':
        muda = jojo.OctJolyne(args.classNumber, in_channels=in_channels)


    print('Criando otimizadores')
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(muda.parameters(),lr=args.learningRate)
    else:
        optimizer = optim.Adam(muda.parameters(), lr=args.learningRate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.8)
    criterion = nn.CrossEntropyLoss().to(device)

    print('Iniciando treino')
    if args.fineTuneWeights is not None:
        print('Loading Pre-trained')
        checkpoint = torch.load(args.fineTuneWeights,map_location=torch.device('cpu'))
        #optimizer.load_state_dict(checkpoint['optimizer'])
        muda.load_state_dict(checkpoint['state_dict'])
        if args.fineTuningClasses > 0:
            nfeats = muda.softmax[-1].in_features
            muda.softmax[-1] = nn.Linear(nfeats, args.fineTuningClasses)

        checkpoint = None

    print(muda)
    muda.to(device)
    cc = SummaryWriter()
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    print(muda)
    for ep in range(args.epochs):
        ibl = ibr = ibtl = ' '
        muda.train()
        lossAcc = []
        totalImages = 0
        for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)

            output, features = muda(currBatch)

            loss = criterion(output, currTargetBatch)
            loss = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())

        lossAvg = sum(lossAcc) / len(lossAcc)
        cc.add_scalar(args.tensorboardname+'/fullData/loss', lossAvg, ep)
        scheduler.step()
        muda.eval()
        total = 0
        correct = 0
        loss_val = []
        with torch.no_grad():
            for data in pro_loader:
                images, labels = data
                outputs, fs = muda(images.to(device))
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels.to(device))
                loss_val.append(loss)

                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

        cResult = correct / total
        tLoss = sum(loss_val) / len(loss_val)
        cc.add_scalar(args.tensorboardname+'/fullData/accuracy', cResult, ep)
        cc.add_scalar(args.tensorboardname+'/fullData/Validation_loss', tLoss, ep)

        state_dict = muda.state_dict()
        opt_dict = optimizer.state_dict()
        fName = '%s_current.pth.tar' % (args.network)
        fName = os.path.join(args.output, fName)
        #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)

        if bestForFoldTLoss > tLoss:
            ibtl = 'X'
            fName = '%s_best_val_loss.pth.tar' % (args.network)
            fName = os.path.join(args.output, fName)
            #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = tLoss

        if bestRankForFold < cResult:
            ibr = 'X'
            fName = '%s_best_rank.pth.tar' % (args.network)
            fName = os.path.join(args.output, fName)
            #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestRankForFold = cResult

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % (args.network)
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        print('[EPOCH %03d] Accuracy of the network on the %d validating images: %.2f %% Training Loss %.5f Validation Loss %.5f - Total of training images %d [%c] [%c] [%c]' % (ep, total, 100 * cResult, lossAvg, tLoss,totalImages,ibl,ibtl,ibr))