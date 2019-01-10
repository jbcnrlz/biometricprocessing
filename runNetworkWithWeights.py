import argparse, torch, torch.optim as optim, torch.nn.functional as F, torch.utils.data, numpy as np
from helper.functions import loadFoldFromFolders, generateImageData

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--loadFromFolder', default=None, help='Load folds from folders', required=True)
    parser.add_argument('--weights', default=None, help='Weights', required=True)
    parser.add_argument('--network', default='networks.PyTorch.jojo.GioGio', help='Network', required=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    argNetwork = args.network.split('.')
    moduleNetwork = '.'.join(argNetwork[:-1])
    networkName = argNetwork[-1]
    net = __import__(moduleNetwork,fromlist=[networkName])
    net = getattr(net, networkName)
    muda = net(52)
    muda.to(device)
    optimizer = optim.SGD(muda.parameters(), lr=0.01, momentum=0.5)
    checkpoint = torch.load(args.weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    muda.load_state_dict(checkpoint['state_dict'])
    muda.eval()

    folds = loadFoldFromFolders(args.loadFromFolder)

    for f, datas in enumerate(folds):
        foldProbe = generateImageData(datas[2],silent=True)
        foldProbeClasses = np.array(datas[3])
        foldProbe = foldProbe / 255

        if np.amin(foldProbeClasses) == 1:
            foldProbeClasses = foldProbeClasses - 1

        foldProbe = torch.from_numpy(np.rollaxis(foldProbe, 3, 1)).float()
        foldProbeClasses = torch.from_numpy(foldProbeClasses)
        pdata = torch.utils.data.TensorDataset(foldProbe, foldProbeClasses)
        test_loader = torch.utils.data.DataLoader(pdata, batch_size=10, shuffle=False)

        total = 0
        correct = 0
        labelsData = [[], []]
        scores = []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = muda(images.to(device))
                scores = scores + np.array(outputs.data).tolist()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
                labelsData[0] = labelsData[0] + np.array(labels).tolist()
                labelsData[1] = labelsData[1] + np.array(predicted).tolist()

        with open('scores_fold_' + str(f) + '.txt', 'w') as ofs:
            for ddv in scores:
                ofs.write(' '.join(list(map(str,ddv))) + '\n')

        with open('labels_fold_' + str(f) + '.txt', 'w') as ofs:
            for ddv in labelsData[0]:
                ofs.write(str(ddv) + '\n')


        cResult = correct / total

        print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * cResult))
