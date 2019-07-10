import torch, math, torch.nn as nn
from torch.nn.functional import conv2d
from torch.autograd.function import Function

class CorrelationImage(nn.Module):
    def __init__(self):
        super(CorrelationImage, self).__init__()

    def forward(self, map1, map2):

        correlation = conv2d(map1,map2)
        correlation = correlation.view(correlation.size(0),-1).diag()
        correlation = correlation / correlation.norm()
        correlation = 1 - correlation
        newResult = map1.clone().fill_(0)
        for i in range(map1.shape[0]):
            newResult[i] = map1[i] + (map2[i] * correlation[i])

        '''

        if map1.is_cuda:
            joininTensor = torch.cuda.FloatTensor(map1.size(0),1).fill_(0)
        else:
            joininTensor = torch.Tensor(map1.size(0),1)

        for i in range(map1.shape[0]):
            joininTensor[i] = conv2d(map1[i].reshape(1,map1.shape[1],map1.shape[2],map1.shape[3]),map2[i].reshape(1,map2.shape[1],map2.shape[2],map2.shape[3]))

        newResult = map1.clone().fill_(0)
        for i in range(map1.shape[0]):
            newResult[i,:,:,:] = (map1[i, :, :, :] * joininTensor[i]) + map2[i, :, :, :]
        '''
        return newResult


class AverageFusion(nn.Module):

    def __init__(self, featuresize, bias=True):
        super(AverageFusion, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(featuresize,featuresize))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(featuresize))
        else:
            self.register_parameter('bias',None)

        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x1, x2):
        bias = self.bias.expand_as(x1)
        x = (x1 + x2) / 2
        return torch.mm(x,self.weight.transpose(0,1)) + bias

class MaxFusion(nn.Module):

    def __init__(self, featuresize, bias=True):
        super(MaxFusion, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(featuresize,featuresize))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(featuresize))
        else:
            self.register_parameter('bias',None)

        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x1, x2):
        bias = self.bias.expand_as(x1)
        x = torch.max(x1,x2)
        return torch.mm(x,self.weight.transpose(0,1)) + bias

class ConcatFusion(nn.Module):

    def __init__(self, featuresize, bias=True):
        super(ConcatFusion, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(featuresize*2,featuresize*2))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(featuresize*2))
        else:
            self.register_parameter('bias',None)

        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x1, x2):
        x = torch.cat((x1,x2),dim=1)
        bias = self.bias.expand_as(x)
        return torch.mm(x,self.weight.transpose(0,1)) + bias