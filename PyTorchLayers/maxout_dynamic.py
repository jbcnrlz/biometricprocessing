import torch
import torch.nn as nn


class MaxoutDynamic(nn.Module):
    def __init__(self, nactive, featsize, scale=1.):
        super(MaxoutDynamic, self).__init__()
        self.nactive_init = nactive
        self.nactive_curr = self.nactive_init
        self.proportion = (featsize - self.nactive_init)/featsize
        self.featsize = featsize
        self.scale = (scale / (1. - self.proportion))

    def forward(self, feat):
        if self.training:
            proportion = (self.featsize - self.nactive_curr) / self.featsize
            scale = 1 / (1. - proportion)

            sortind = torch.argsort(feat, descending=False).to(feat.get_device())
            m = torch.Tensor(self.featsize * torch.Tensor((range(feat.size()[0])))).repeat((self.featsize, 1)).t().long().to(feat.get_device())
            m2 = sortind + m
            m2 = m2[:, range(int(feat.size()[1] - self.nactive_curr))]

            batchsize = feat.size()[0]
            feat = feat.view(feat.size()[0] * self.featsize).clone()
            feat[m2] = 0
            feat *= scale
            feat = feat.view(batchsize, self.featsize)
        return feat