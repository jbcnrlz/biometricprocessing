from torchvision import transforms
from datasetClass.structures import loadDatasetFromFolder
import networks.PyTorch.vgg_face_dag as vgg, argparse, numpy as np, torch, torch.optim as optim
import torch.utils.data, shutil, os
from tensorboardX import SummaryWriter
from helper.functions import saveStatePytorch, shortenNetwork, plotFeaturesCenterloss
import torch.nn as nn
from PyTorchLayers.center_loss import compute_center_loss, get_center_delta, CenterLoss

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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.562454871481894, 0.8208898956471341, 0.395364053852456],
                             std=[0.43727472598867456, 0.31812502566122625, 0.3796120355707891])
    ])

    folds = loadDatasetFromFolder(args.pathBase,validationSize='auto',transforms=dataTransform)

    gal_loader = torch.utils.data.DataLoader(folds[0], batch_size=args.batch, shuffle=False)
    pro_loader = torch.utils.data.DataLoader(folds[1], batch_size=args.batch, shuffle=False)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    print('Criando diretorio')
    os.makedirs(args.output)
    muda = vgg.vgg_face_dag_load()
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
    closs = CenterLoss(args.classes,2622).to(device)
    optim_closs = optim.SGD(closs.parameters(), lr=0.5)
    alpha = 0.5
    optimizer = optim.SGD(muda.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer,20,gamma=0.8)
    criterion = nn.CrossEntropyLoss().to(device)
    print('Iniciando treino')

    cc = SummaryWriter()
    bestForFold = 500000
    bestRankForFold = -1
    ep = 0
    while True:
        muda.train()
        scheduler.step()
        lossAcc = []
        galleryFeatures = []
        galleryClasses = []

        for bIdx, (currBatch, currTargetBatch) in enumerate(gal_loader):

            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)
            features, output = muda(currBatch)

            loss = criterion(output, currTargetBatch)
            loss = loss + (alpha * closs(currTargetBatch,features))

            optimizer.zero_grad()
            optim_closs.zero_grad()

            loss.backward()

            optimizer.step()
            optim_closs.step()

            galleryFeatures.append(features.data.cpu().numpy())
            galleryClasses.append(currTargetBatch.data.cpu().numpy())
            lossAcc.append(loss.item())

        lossAvg = sum(lossAcc) / len(lossAcc)
        cc.add_scalar('VGG/loss', lossAvg, ep)

        featurescloss = plotFeaturesCenterloss(np.concatenate(galleryFeatures,0),np.concatenate(galleryClasses,0),args.classes)
        cc.add_figure('VGG/features/train', featurescloss, ep)

        muda.eval()
        total = 0
        correct = 0
        labelsData = [[],[]]
        lossV = []
        probeFeatures = []
        probeClasses = []

        with torch.no_grad():
            for data in pro_loader:
                images, labels = data
                fs, outputs = muda(images.to(device))
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.to(device) ).sum().item()
                labelsData[0] = labelsData[0] + np.array(labels).tolist()
                labelsData[1] = labelsData[1] + np.array(predicted).tolist()
    
            probeFeatures.append(fs.data.cpu().numpy())
            probeClasses.append(labels.data.cpu().numpy())


        cResult = correct / total
        featurescloss = plotFeaturesCenterloss(np.concatenate(probeFeatures,0),np.concatenate(probeClasses,0),args.classes)
        cc.add_figure('VGG/features/test', featurescloss, ep)

        cc.add_scalar('VGG/accuracy', cResult, ep)

        print('[EPOCH %d] Accuracy of the network on the %d test images: %.2f%% Loss %f' % (ep,total,100 * cResult,lossAvg))
        
        print('Salvando epoch atual')
        state_dict = muda.state_dict()
        opt_dict = optimizer.state_dict()
        fName = '%s_current.pth.tar' % ('vgg')
        fName = os.path.join(args.output, fName)
        saveStatePytorch(fName, state_dict, opt_dict, ep + 1)

        if bestRankForFold < cResult:
            print('Salvando melhor rank')
            fName = '%s_best_rank.pth.tar' % ('vgg')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestRankForFold = cResult

        if bestForFold > lossAvg:
            print('Salvando melhor Loss')
            fName = '%s_best_loss.pth.tar' % ('vgg')
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