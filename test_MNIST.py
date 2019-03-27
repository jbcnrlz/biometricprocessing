import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from PyTorchLayers.center_loss import CenterLoss, DensityLoss, ICenterLoss
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np
from scipy.spatial.distance import euclidean

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*3*3, 2)
        self.ip2 = nn.Linear(2, 10, bias=False)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*3*3)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        return ip1, F.log_softmax(ip2, dim=1)

def visualize(feat, labels, epoch, centers):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])

    for i in range(10):
        plt.plot(centers[i,0],centers[i,1],'.',c='#000000')
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    plt.xlim(left=-20,right=20)
    plt.ylim(bottom=-20,top=20)
    plt.text(-19.8,18,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)

def distancebetweencenters(centers):
    distance = np.zeros((centers.shape[0],centers.shape[0]))
    for i, c1 in enumerate(centers):
        for j, nb in enumerate(centers):
            distance[i][j] = euclidean(c1,nb)

    print(distance)

def train(epoch):
    ip1_loader = []
    idx_loader = []
    lossAcc = []
    for i,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        ip1, pred = model(data)
        loss = nllloss(pred, target) + loss_weight * centerloss(target, ip1) - 0.5 * mloss(centerloss.centers,ip1,target)

        optimizer4nn.zero_grad()
        optimzer4center.zero_grad()
        #optimzer4I.zero_grad()

        loss.backward()
        #dloss.backward()

        optimizer4nn.step()
        optimzer4center.step()
        #optimzer4I.step()

        ip1_loader.append(ip1)
        idx_loader.append((target))

        lossAcc.append(loss.item())

    return sum(lossAcc) / len(lossAcc)

def test(epoch,loader,lossVal):
    total=0
    correct=0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            fs, outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    cResult = correct / total

    print('[EPOCH %03d] Accuracy of the network on the %d test images: %.2f%% - Loss %f' % (epoch,total,100 * cResult,lossVal))

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    # Dataset
    trainset = datasets.MNIST('../MNIST', download=True,train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))

    idxsVal = random.sample(list(range(len(trainset))),int(len(trainset) * 0.1))
    idxsTrain = [i for i in range(len(trainset)) if i not in idxsVal]

    train_loader = Subset(trainset,idxsTrain)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    val_loader = Subset(trainset,idxsVal)
    val_loader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)

    # Model
    model = Net().to(device)

    # NLLLoss
    nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
    # CenterLoss
    loss_weight = 1
    centerloss = CenterLoss(10, 2).to(device)
    icloss = ICenterLoss(10, 2).to(device)
    # optimzer4nn
    optimizer4nn = optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
    sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)
    mloss = DensityLoss().to(device)
    # optimzer4center
    optimzer4center = optim.SGD(centerloss.parameters(), lr =0.5)

    optimzer4I = optim.SGD(icloss.parameters(), lr=0.0001)
    cc = SummaryWriter()
    for epoch in range(100):
        sheduler.step()
        # print optimizer4nn.param_groups[0]['lr']
        lossVal = train(epoch+1)
        test(epoch+1,val_loader,lossVal)

    distancebetweencenters(centerloss.centers.data.cpu())


    testset = datasets.MNIST('../MNIST', download=True,train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    testLoader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=4)
    test(1,testLoader,0)
