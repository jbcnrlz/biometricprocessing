import networks.PyTorch.jojo as jojo, argparse, torch.optim as optim
import torch.utils.data, shutil, os
from helper.functions import saveStatePytorch
from torch.utils.tensorboard import SummaryWriter
from datasetClass.structures import loadDatasetFromFolder, loadFoldsDatasets, loadFoldsDatasetsDepthDI
from torchvision import transforms, models
import torch.nn as nn

def initialize_model(num_classes,channels=3,modelName='resnet'):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    if modelName == 'resnet':
        model_ft = models.resnet50(pretrained=False)
        if channels > 3:
                model_ft.conv1 = nn.Conv2d(channels,64,kernel_size=7,stride=2,padding=3,bias=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif modelName == 'inception':
        model_ft = models.inception_v3(pretrained=False)
        if channels > 3:
                model_ft.Conv2d_1a_3x3.conv = nn.Conv2d(channels,32,kernel_size=3,stride=2,bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif modelName == 'mobilenet':
        model_ft = models.mobilenet_v3_large(pretrained=False)
        if channels > 3:
                model_ft.features[0][0] = nn.Conv2d(channels,16,kernel_size=3,stride=2,bias=False)
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    input_size = 100

    return model_ft, input_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-p', '--pathBase', default='generated_images_lbp_frgc', help='Path for faces', required=False)
    parser.add_argument('-b', '--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('-c', '--classNumber', type=int, default=466, help='Quantity of classes', required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to be run', required=False)
    parser.add_argument('--output', default=None, help='Output Folder', required=False)
    parser.add_argument('--extension', help='Extension from files', required=False, default='png')
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--tensorboardname', help='Learning Rate', required=False, default='GioGioFullTraining')
    parser.add_argument('--optimizer', help='Optimizer', required=False, default="sgd")
    parser.add_argument('--depthFolder', help='Folder with the depth', required=False, default=None)
    parser.add_argument('--meanImage', help='Mean image', nargs='+', required=False, type=float)
    parser.add_argument('--stdImage', help='Std image', nargs='+', required=False, type=float)
    parser.add_argument('--model', help='Model to finetune', required=False, default='resnet')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.meanImage,
                             std=args.stdImage)
    ])

    print('Carregando dados')
    if args.depthFolder is None:
        if os.path.exists(os.path.join(args.pathBase,'1')):
            folds = loadFoldsDatasets(args.pathBase, dataTransform)[0]
        else:
            folds = loadDatasetFromFolder(args.pathBase, validationSize='auto', transforms=dataTransform)
    else:
        depthTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6928382911600398],std=[0.18346924017986496])
        ])
        folds = loadFoldsDatasetsDepthDI(args.pathBase, dataTransform, args.depthFolder,depthTransform)[0]

    gal_loader = torch.utils.data.DataLoader(folds[0], batch_size=args.batch, shuffle=True)
    pro_loader = torch.utils.data.DataLoader(folds[1], batch_size=args.batch, shuffle=False)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    print('Criando diretorio')
    os.makedirs(args.output)
    if args.depthFolder is None:
        channelsForImage = 3 if args.extension != 'png' else 4
    else:
        channelsForImage = 5

    muda, sizeim = initialize_model(args.classNumber,channelsForImage,modelName=args.model)

    print('Criando otimizadores')
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(muda.parameters(),lr=args.learningRate)
    else:
        optimizer = optim.Adam(muda.parameters(), lr=args.learningRate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.8)
    criterion = nn.CrossEntropyLoss().to(device)

    print('Iniciando treino')

    print(muda)
    muda.to(device)
    print(device)
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
            if args.optimizer is None:
                totalImages += currBatch.shape[0]
                currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)

                output = muda(currBatch)

                loss = criterion(output, currTargetBatch)
                loss = loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lossAcc.append(loss.item())
            else:
                currBatch, depthBatch = currBatch
                totalImages += currBatch.shape[0]
                currTargetBatch, currBatch, depthBatch = currTargetBatch.to(device), currBatch.to(device), depthBatch.to(device)

                output = muda(torch.cat((currBatch,depthBatch[:,:1,:,:]),1))

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
                if args.optimizer is None:
                    images, labels = data
                    outputs = muda(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels.to(device))
                    loss_val.append(loss)

                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
                else:
                    images, labels = data
                    cbt, cdb = images
                    cbt, cdb = cbt.to(device), cdb.to(device)
                    outputs = muda(torch.cat((cbt,cdb[:,:1,:,:]),1))
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
        fName = '%s_current.pth.tar' % ('resnet50')
        fName = os.path.join(args.output, fName)
        #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)

        if bestForFoldTLoss > tLoss:
            ibtl = 'X'
            fName = '%s_best_val_loss.pth.tar' % ('resnet50')
            fName = os.path.join(args.output, fName)
            #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = tLoss

        if bestRankForFold < cResult:
            ibr = 'X'
            fName = '%s_best_rank.pth.tar' % ('resnet50')
            fName = os.path.join(args.output, fName)
            #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestRankForFold = cResult

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % ('resnet50')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        print('[EPOCH %03d] Accuracy of the network on the %d validating images: %.2f %% Training Loss %.5f Validation Loss %.5f - Total of training images %d [%c] [%c] [%c]' % (ep, total, 100 * cResult, lossAvg, tLoss,totalImages,ibl,ibtl,ibr))