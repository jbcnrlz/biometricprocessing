from torchvision import transforms
from datasetClass.structures import loadDatasetFromFolder
import networks.PyTorch.vgg_face_dag as vgg, argparse, numpy as np, torch, torch.optim as optim
import torch.utils.data, shutil, os
from tensorboardX import SummaryWriter
from helper.functions import saveStatePytorch, shortenNetwork, generateRandomColors
import torch.nn as nn

def save_checkpoint(state,is_best,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-p', '--pathBase', default='generated_images_lbp_frgc', help='Path for faces', required=False)
    parser.add_argument('-b', '--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to be run', required=False)
    parser.add_argument('--fineTuneWeights', help='Do fine tuning with weights', required=True)
    parser.add_argument('--output', default=None, help='Output Folder', required=True)
    parser.add_argument('--extension', help='Extension from files', required=False, default='png')
    parser.add_argument('--classes', help='Quantity of classes', required=False, default=458, type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Carregando dados')

    dataTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    folds = loadDatasetFromFolder(args.pathBase,validationSize='auto',transforms=dataTransform)

    gal_loader = torch.utils.data.DataLoader(folds[0], batch_size=args.batch, shuffle=True)
    pro_loader = torch.utils.data.DataLoader(folds[1], batch_size=args.batch, shuffle=False)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    print('Criando diretorio')
    os.makedirs(args.output)
    muda = vgg.vgg_face_dag_load(weights_path=args.fineTuneWeights)
    muda = vgg.vgg_smaller(muda,args.classes)
    shortenedList = shortenNetwork(
        list(muda.convolutional.children()),
        [0, 1, 4, 5, 6, 9, 10, 11, 16, 17, 18, 23],True
    )
    muda.convolutional = nn.Sequential(*shortenedList)
    muda.fullyConnected = nn.Sequential(
        nn.Linear(in_features=18432, out_features=4096),
        *list(muda.fullyConnected.children())[1:]
    )
    muda.to(device)
    print(muda)

    print('Criando otimizadores')
    alpha = 1
    optimizer = optim.SGD(muda.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer,20,gamma=0.8)
    criterion = nn.CrossEntropyLoss().to(device)

    print('Iniciando treino')

    cc = SummaryWriter()
    bestForFold = 500000
    bestRankForFold = -1
    ep = 0
    colors = generateRandomColors(args.classes)
    while True:
        ibl = ibr = ' '
        muda.train()
        scheduler.step()
        lossAcc = []
        galleryFeatures = []
        galleryClasses = []

        for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):
            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)
            features, output = muda(currBatch)

            loss = criterion(output, currTargetBatch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            galleryFeatures.append(features.data.cpu().numpy())
            galleryClasses.append(currTargetBatch.data.cpu().numpy())
            lossAcc.append(loss.item())

        lossAvg = sum(lossAcc) / len(lossAcc)
        cc.add_scalar('VGG/loss', lossAvg, ep)

        muda.eval()
        total = 0
        correct = 0
        labelsData = [[],[]]
        lossV = []
        probeFeatures = []
        probeClasses = []
        loss_val = []
        with torch.no_grad():
            for data in pro_loader:
                images, labels = data
                fs, outputs = muda(images.to(device))
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels.to(device))
                loss_val.append(loss)

                total += labels.size(0)
                correct += (predicted == labels.to(device) ).sum().item()
                labelsData[0] = labelsData[0] + np.array(labels.cpu()).tolist()
                labelsData[1] = labelsData[1] + np.array(predicted.cpu()).tolist()
    
                probeFeatures.append(fs.data.cpu().numpy())
                probeClasses.append(labels.data.cpu().numpy())


        cResult = correct / total
        tLoss = sum(loss_val) / len(loss_val)
        cc.add_scalar('VGG/accuracy', cResult, ep)
        state_dict = muda.state_dict()
        opt_dict = optimizer.state_dict()
        fName = '%s_current.pth.tar' % ('vgg')
        fName = os.path.join(args.output, fName)
        saveStatePytorch(fName, state_dict, opt_dict, ep + 1)

        if bestRankForFold < cResult:
            ibr = 'X'
            fName = '%s_best_rank.pth.tar' % ('vgg')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestRankForFold = cResult

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % ('vgg')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        print('[EPOCH %03d] Accuracy of the network on the %d validating images: %03.2f %% Training Loss %.5f Validation Loss %.5f [%c] [%c]' % (ep, total, 100 * cResult, lossAvg, tLoss, ibl, ibr))
        cc.add_scalar('VGG/validation_loss', tLoss, ep)

        ep += 1
        if args.epochs < 0:
            if lossAvg < 0.0001:
                break
        else:
            args.epochs -= 1
            if args.epochs < 0:
                break

    print('Terminou')