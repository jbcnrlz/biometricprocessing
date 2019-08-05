from PyTorchLayers.maxout_dynamic import *
from PyTorchLayers.octoconv import *
from PyTorchLayers.CorrelationImages import *

class FusingNetwork(nn.Module):
    def __init__(self,featureSize,classes):
        super(FusingNetwork,self).__init__()

        self.classifier = nn.Sequential(
            MaxoutDynamic(featureSize,  featureSize),
            nn.Dropout(),
            nn.Linear(featureSize, 2048),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            MaxoutDynamic(1024, 2048),
            nn.Linear(2048, classes)
        )

        self.avgFuse = AverageFusion(featureSize)
        #self.maxFuse = MaxFusion(featureSize)
        #self.catFuse = ConcatFusion(featureSize)

    def forward(self, x):
        #outFeatures = self.catFuse(self.avgFuse(x[0].view(x[0].size(0), -1),x[1].view(x[1].size(0), -1)), self.maxFuse(x[0].view(x[0].size(0), -1),x[1].view(x[1].size(0), -1)))
        outFeatures = self.avgFuse(x[0].view(x[0].size(0), -1), x[1].view(x[1].size(0), -1))
        outFeatures = self.classifier(outFeatures)
        return  self.softmax(outFeatures), outFeatures


def conv8x8(in_planes, out_planes, stride=4):
    return nn.Conv2d(in_planes, out_planes, kernel_size=8, stride=stride,padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes):
        super(BasicBlock, self).__init__()
        self.conv1 = conv8x8(inplanes,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(64, 128)
        self.bn2 = nn.BatchNorm2d(128)

    def addResidualInformation(self,out,residualOv,residualUn,resize=True):
        residualSum =None
        if resize:
            max_redux = nn.MaxPool2d(4,stride=4)
            identityOv = max_redux(residualOv)[:,1:23,1:23]
            identityUnd = max_redux(residualUn)[:,1:23,1:23]

            residualSum = identityOv+identityUnd
        else:
            residualSum = residualOv + residualUn

        for i in range(out.shape[1]):
            out[:,i,:,:] += residualSum

        return out

    def forward(self, x):
        identityOv = x[:, 4,:,:]
        identityUnd = x[:, 5, :, :]
        data = x[:, 0:4, :, :]
        data= self.addResidualInformation(data, identityOv, identityUnd,False)
        out = self.conv1(data)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)
        out = nn.AvgPool2d(kernel_size=3, stride=2)(out)
        return out

class Joseph(nn.Module):
    def __init__(self,classes):
        super(Joseph, self).__init__()
        self.features = BasicBlock(4)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128*10*10, classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), 128*10*10)
        out = self.classifier(out)
        return out

class OctJolyne(nn.Module):

    def __init__(self,classes,imageInput=(100,100),in_channels=4):
        self.imageInput = imageInput
        super(OctJolyne,self).__init__()
        self.features = nn.Sequential(
            OctConv(in_channels, 256, kernel_size=8, stride=4,alphas=[0,0.5]),
            nn.ReLU(inplace=True),
            OctConv(256, 512, kernel_size=4,stride=2),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            OctConv(512, 1024, kernel_size=2, stride=1,alphas=[0.5,0]),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024*10*10, 2048),
            nn.ReLU(inplace=True),
            MaxoutDynamic(1024, 2048),
            nn.Dropout(),
            nn.Linear(2048, 2048),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            MaxoutDynamic(1024, 2048),
            nn.Linear(2048, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return  self.softmax(x), x

class SyameseJolyne(nn.Module):

    def normedCrossCorrelation(self, a, b):
        correlationBetData = ((a - torch.mean(a)) * (b - torch.mean(b))) / (torch.sqrt(torch.var(a) * torch.var(b)))
        return correlationBetData

    def __init__(self,classes,imageInput=(100,100)):
        self.imageInput = imageInput
        super(SyameseJolyne,self).__init__()

        #self.input3 = nn.Conv2d(3, 256, kernel_size=8, stride=3)

        self.features = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=8, stride=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(512, 1024, kernel_size=2, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )

        self.classifier = nn.Sequential(
            MaxoutDynamic(int(16384 / 2),  16384),
            nn.Dropout(),
            nn.Linear(16384, 2048),
            nn.ReLU(inplace=True),
            MaxoutDynamic(1024, 2048),
            nn.Dropout(),
            nn.Linear(2048, 2048),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            MaxoutDynamic(1024, 2048),
            nn.Linear(2048, classes)
        )

        self.avgFuse = AverageFusion(16384)
        #self.maxFuse = MaxFusion(16384)
        #self.catFuse = ConcatFusion(9216)

    def forward(self, x):
        '''
        outFeat = self.input4(x[0])
        outFeat = self.features(outFeat)
        outFeat = outFeat.view(outFeat.size(0), -1)
        outFeat = self.classifier(outFeat)
        '''

        outFeatures = []
        for i in x:
            outFeat = self.features(i)
            outFeat = outFeat.view(outFeat.size(0), -1)
            #outFeat = self.classifier(outFeat)
            outFeatures.append(outFeat)

        #outFeatures = self.activationJoin(self.joinmaps(outFeatures[0],outFeatures[1]))
        #outFeatures = self.avgFuse(outFeatures[0],outFeatures[1])
        #outFeatures = self.maxFuse(outFeatures[0], outFeatures[1])
        #outFeatures = self.catFuse(self.avgFuse(outFeatures[0],outFeatures[1]), self.maxFuse(outFeatures[0],outFeatures[1]))
        outFeatures = self.avgFuse(outFeatures[0], outFeatures[1])
        outFeatures = self.classifier(outFeatures)
        return  self.softmax(outFeatures), outFeatures


class Jolyne(nn.Module):

    def __init__(self,classes,imageInput=(100,100),in_channels=4):
        self.imageInput = imageInput
        super(Jolyne,self).__init__()

        #self.savBlock = nn.Conv2d(in_channels,1792,stride=10,kernel_size=10)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=8, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=2, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=10, stride=2, padding=2)
        self.maxpoolb2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.reducepool = nn.MaxPool2d(kernel_size=2, stride=2)
        '''
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=8, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(512, 1024, kernel_size=2, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2)
        )
        '''
        self.classifier = nn.Sequential(
            nn.Dropout(),
            MaxoutDynamic(int(1792 / 2),1792),
            nn.Linear(1792, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            MaxoutDynamic(512, 1024),
            nn.Linear(1024, 1024),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            MaxoutDynamic(512,1024),
            nn.Linear(1024, classes)
        )

    def forward(self, x):
        #savWs = self.savBlock(x)
        x = self.block1(x)
        mp1 = self.maxpool(x)
        x = self.block2(x)
        mp2 = self.maxpoolb2(x)
        x = self.block3(x)
        x = torch.cat((x,mp1,mp2),dim=1)
        '''
        fv = x.clone()[:,:,1,1]
        savWs = self.savBlock.weight.data.mean(dim=1)
        savWs = savWs.view(savWs.shape[0], savWs.shape[1] * savWs.shape[2], 1)
        for i in range(fv.shape[0]):
            t2 = x[i, :, :, :]
            t2 = t2.view(x.shape[1],x.shape[2] * x.shape[3],1).transpose(1,2)
            fv[i] = torch.bmm(t2,savWs).flatten()

        fv = self.classifier(fv)
        '''
        #x = self.reducepool(x)
        #x = x.view(x.size(0),-1)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = self.classifier(x)
        return  self.softmax(x), x

class GioGio(nn.Module):

    def calculateSize(self,dim,layer,inputSize):
        padding = layer.padding if (type(layer.padding) is not list) else layer.padding[dim]
        dilation = layer.dilation if (type(layer.dilation) is not list) else layer.dilation[dim]
        kernel_size = layer.kernel_size if (type(layer.kernel_size) is not list) else layer.kernel_size[dim]
        stride = layer.stride if (type(layer.stride) is not list) else layer.stride[dim]
        return int(((inputSize+(padding*2)-dilation*(kernel_size-1)-1) / stride) + 1)

    def __init__(self,classes,imageInput=(100,100),in_channels=4):
        self.imageInput = imageInput
        super(GioGio,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(384*5*5, 2048),
            nn.ReLU(inplace=True),
            MaxoutDynamic(1024, 2048),
            nn.Dropout(),
            nn.Linear(2048, 2048),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            MaxoutDynamic(1024, 2048),
            nn.Linear(2048, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return  self.softmax(x), x