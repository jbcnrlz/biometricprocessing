from helper.functions import generateData, generateFoldsOfData, generateImageData, loadFoldFromFolders
import networks.PyTorch.jojo as jojo, argparse, numpy as np, torch, torch.optim as optim, torch.nn.functional as F
import torch.utils.data, shutil, os, time

def save_checkpoint(state,is_best,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.loadFromFolder is None:
        imageData, classesData = generateData(args.pathBase)
        folds = generateFoldsOfData(args.folds,imageData,classesData)

    else:
        folds = loadFoldFromFolders(args.loadFromFolder)

    if not os.path.exists('training_pytorch'):
        os.makedirs('training_pytorch')

    foldResults = []
    trainTimeFolds = []
    for f, datas in enumerate(folds):
        print('====================== Fazendo Fold =======================')
        muda = jojo.GioGio(args.classNumber)
        print(muda)
        muda.to(device)

        foldResults.append([])
        foldProbe = generateImageData(datas[2])
        foldProbeClasses = np.array(datas[3])
        foldGallery = generateImageData(datas[0])
        foldGalleryClasses = np.array(datas[1])

        foldGallery = foldGallery / 255
        foldProbe = foldProbe / 255
        if np.amin(foldGalleryClasses) == 1:
            foldGalleryClasses = foldGalleryClasses - 1
            foldProbeClasses = foldProbeClasses - 1

        muda.train()
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

        if not os.path.exists(os.path.join('training_pytorch', str(f))):
            os.makedirs(os.path.join('training_pytorch', str(f)))

        if args.fineTuneWeights is not None:
            checkpoint = torch.load(args.fineTuneWeights)
            optimizer.load_state_dict(checkpoint['optimizer'])
            muda.load_state_dict(checkpoint['state_dict'])

        bestResult = -1
        bestEpoch = -1

        fTrainignTime = []
        for ep in range(args.epochs):

            start_time = time.time()
            for bIdx, (currBatch,currTargetBatch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = muda(currBatch.to(device))
                loss = F.cross_entropy(output, currTargetBatch.to(device))
                loss.backward()
                optimizer.step()
                print('[%d, %05d de %05d] loss: %.3f' %(ep+ 1, bIdx + 1, qntBatches, loss.item()))
            finishTime = time.time() - start_time
            fTrainignTime.append(finishTime)

            total = 0
            correct = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    outputs = muda(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device) ).sum().item()

            cResult = correct / total

            print('[EPOCH %d] Accuracy of the network on the %d test images: %d %%' % (ep,total,100 * cResult))

            if bestResult < cResult:
                bestResult = cResult
                bestEpoch = ep

            if ep % 10 == 0:
                fName = '%s_checkpoint_%05d.pth.tar' % ('GioGio',ep)
                fName = os.path.join('training_pytorch', str(f),fName)
                save_checkpoint({
                    'epoch': ep + 1,
                    'arch': 'GioGio',
                    'state_dict': muda.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, False, fName)
                foldResults[-1].append(correct / total)

        trainTimeFolds.append(sum(fTrainignTime))
        print('Best result %2.6f Epoch %d' % (bestResult*100,bestEpoch))


    print(foldResults)
    print(trainTimeFolds)