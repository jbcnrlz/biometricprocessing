import torch
import torch.nn as nn
from torch.autograd.function import Function

def testFuckingloss(centers,labels):
    centerDist = centers.new_empty(centers.size()[0])
    for i in range(centers.size()[0]):
        for j in range(centers.size()[0]):
            if i != j:
                centerDist[i] += torch.dist(centers[j],centers[i])**2

        centerDist[i] = torch.sum(centerDist[i]) / (centers.size()[0] - 1)
    return (centerDist.mean() / centers.std()) / labels.size(0)

class DensityLoss(nn.Module):
    def __init__(self):
        super(DensityLoss, self).__init__()
        self.densitylossfunc = DensityLossFunc.apply

    def forward(self, centers, features, label):
        loss = self.densitylossfunc(centers, features, label)
        return loss


class DensityLossFunc(Function):
    @staticmethod
    def forward(ctx, centers, features, labels):
        ctx.save_for_backward(centers, features, labels)
        centerDist = centers.new_empty(centers.size()[0])
        for i in range(centers.size()[0]):
            for j in range(centers.size()[0]):
                if i != j:
                    centerDist[i] += (centers[j] - centers[i]).pow(2).sum()

            centerDist[i] = torch.sum(centerDist[i]) / (centers.size(0) - 1)
        return (centerDist.sum() / centers.size(0)) / centerDist.var() / labels.size(0)

    @staticmethod
    def backward(ctx, grad_output):

        centers, features, labels  = ctx.saved_tensors
        labels = labels.long()
        gradients = features.new_zeros(features.size())
        average_centers = centers.new_zeros(centers.size())
        for i in range(centers.size(0)):
            for j in range(centers.size(0)):
                average_centers[i] += 2*(centers[j] - centers[i])

        avCenter = centers.mean()
        for i in range(features.size(0)):
            gradients[i] = ((2 * (centers[labels[i]] - avCenter) * (features[i] - centers[labels[i]])) / 2 * (centers.size(0))) * average_centers[labels[i]]

        return None, 0.0001*(-grad_output * gradients) / labels.size(0) , None


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


class ICenterLoss(nn.Module):
    def __init__(self,num_classes,features):
        super(ICenterLoss, self).__init__()
        self.icenters = nn.Parameter(torch.randn(num_classes,features))
        self.icenterlossfunc = ICenterLossFunc.apply

    def forward(self, centers, features, label):
        loss = self.icenterlossfunc(centers, label, features, self.icenters)
        return loss

class ICenterLossFunc(Function):
    @staticmethod
    def forward(ctx, centers, labels, features, icenters):
        ckEucNorm = 0.2 * torch.sum(icenters.pow(2).sum(dim=1)) / icenters.size(0)
        cys = centers.norm(p=2)**2
        newCenters = ckEucNorm - cys
        ctx.save_for_backward(centers, labels,features, icenters)
        return newCenters**2 / labels.size(0)

    @staticmethod
    def backward(ctx, grad_output):

        centers, labels,features, icenters = ctx.saved_tensors
        batchSize = labels.size(0)
        newCenters = icenters.new_zeros(icenters.size())

        #calculating center dispersion
        ones = centers.new_ones(labels.size(0))
        counts = centers.new_zeros(centers.size(0))
        counts = counts.scatter_add_(0, labels.long(), ones)
        counts[counts == 0] = 1
        newCenters.scatter_add(0, labels.unsqueeze(1).expand(features.size()), features)
        newCenters = newCenters / counts.view(-1, 1)

        # calculating gradient
        bcount = labels.bincount(minlength=centers.size(0)).index_select(0, labels.long())
        centers_batch = centers.index_select(0, labels.long())
        ckEucNorm = torch.sum(icenters.pow(2).sum(dim=1)) / icenters.size(0)
        cys = centers_batch.pow(2).sum(dim=1)
        gradients = (1/bcount.float()).view(-1,1) * (0.2 * ckEucNorm - cys.view(-1,1)) * ((4*0.2/centers.size(0)) - 4)*centers_batch

        return None, None, -grad_output * gradients / batchSize, newCenters / batchSize

