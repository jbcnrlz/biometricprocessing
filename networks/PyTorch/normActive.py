import torch
import torch.nn as nn

def smart_sort(x, permutation):
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret


class NormActive(nn.Module):
    def __init__(self, nactive, featsize, scale=1.):
        super(NormActive, self).__init__()
        self.nactive = nactive
        self.proportion = (featsize - self.nactive)/featsize
        self.featsize = featsize
        self.scale = (scale / (1. - self.proportion))

    def forward(self, feat):

        sortind = torch.argsort(feat, descending=False).to("cuda:0")
        m = torch.Tensor(self.featsize * torch.Tensor((range(feat.size()[0])))).repeat((self.featsize, 1)).t().long().to("cuda:0")
        m2 = sortind + m
        m2 = m2[:, range(int(feat.size()[1] - self.nactive))]

        batchsize = feat.size()[0]
        feat = feat.view(feat.size()[0] * self.featsize)
        feat[m2] = 0
        feat *= self.scale
        feat = feat.view(batchsize, self.featsize)

        return feat


class NormActiveAdaptDropWithLoss(nn.Module):
    def __init__(self, featsize):
        super(NormActiveAdaptDropWithLoss, self).__init__()

        self.featsize = featsize
        self.scale = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.scale, 1.)

    def forward(self, feat, prop):

        scale = self.scale/prop
        toKeep = torch.round(self.featsize * prop)
        sortind = torch.argsort(feat, descending=False, dim=1).to("cuda:0")

        mask = torch.zeros_like(feat)

        for i in range(feat.size()[0]):
            mask[i, sortind[i, range(int(toKeep[i]))]] = 1.

        scale = scale.view(-1, 1).repeat(1, self.featsize).to("cuda:0")
        feat *= (mask*scale)

        return feat
