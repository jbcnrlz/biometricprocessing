import torch
import torch.nn as nn
from torch.nn.functional import conv2d

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

