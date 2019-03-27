import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
class Earings(nn.Module):
    def __init__(self,network1,network2,blockNet1=['features'],blockNet2=['convolutional','view','fullyConnected']):
        super(Earings, self).__init__()
        n1 = dict(network1.named_children())
        self.net1 = nn.Sequential(OrderedDict(n1))

        n2 = dict(network2.named_children())
        self.net2 = nn.Sequential(OrderedDict(n2))

        self.blockNet1 = blockNet1
        self.blockNet2 = blockNet2
        for i in range(len(n1[self.blockNet1[-1]])):
            try:
                outFeatures = n1[self.blockNet1[-1]][-(1+i)].out_channels
                break
            except:
                continue
        inFeatures = n2[self.blockNet2[0]][0].in_channels
        self.weld=nn.Conv2d(outFeatures,inFeatures,kernel_size=3)
        self.weldFully = nn.Linear(12800,4096)

    def forward(self, x):
        for b in self.blockNet1:
            if b == 'view':
                x = x.view(x.size(0), -1)
            else:
                x = self.net1._modules[b](x)
        x = self.weld(x)
        for b in self.blockNet2:
            if b == 'view':
                x = x.view(x.size(0), -1)
                if self.weldFully is not None:
                    x = self.weldFully(x)
            else:
                x = self.net2._modules[b](x)

        return x, self.net2.softmax(x)