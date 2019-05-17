from PyTorchLayers.maxout_dynamic import *
from PyTorchLayers.octoconv import *
from PyTorchLayers.CorrelationImages import *
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

    def __init__(self,classes,channelsForImages,imageInput=(100,100)):
        self.imageInput = imageInput
        self.channels = channelsForImages
        super(SyameseJolyne,self).__init__()

        self.input3 = nn.Conv2d(3, 256, kernel_size=8, stride=4)
        self.input4 = nn.Conv2d(4, 256, kernel_size=8, stride=4)

        self.features = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=2, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 512, kernel_size=2,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(512, 768, kernel_size=2, stride=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25600, 2048),
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

        #self.joinmaps = CorrelationImage()
        #self.activationJoin = nn.ReLU(inplace=True)

        self.onedpull = nn.AvgPool1d(3)

    def forward(self, x):

        outFeatures = []
        for i in x:
            if i.shape[1] == 3:
                outFeat = self.input3(i)
            else:
                outFeat = self.input4(i)
            outFeat = self.features(outFeat)
            outFeatures.append(outFeat)

        #outFeatures = self.activationJoin(self.joinmaps(outFeatures[0],outFeatures[1]))
        outFeatures1 = outFeatures[0].view(outFeatures[0].size(0), -1)
        outFeatures2 = outFeatures[1].view(outFeatures[1].size(0), -1)
        outFeatures = self.normedCrossCorrelation(outFeatures1,outFeatures2)
        idx = np.ogrid[tuple(map(slice, outFeatures.shape))]
        ords = outFeatures.argsort(dim=1)
        idx[1] = ords
        size = int(outFeatures1.shape[1] / 2)
        outFeatures1 = outFeatures1[idx]
        outFeatures2 = outFeatures2[idx]
        #outFeatures = outFeatures1 + (outFeatures2 * outFeatures)
        outFeatures = torch.cat((outFeatures1[:,0:size],outFeatures2[:,0:size]),1)
        #outFeatures = self.onedpull(outFeatures.reshape((outFeatures.shape[0],1,outFeatures.shape[1]))).reshape((outFeatures.shape[0],-1))
        outFeatures = self.classifier(outFeatures)
        return  self.softmax(outFeatures), outFeatures


class Jolyne(nn.Module):

    def __init__(self,classes,imageInput=(100,100),in_channels=4):
        self.imageInput = imageInput
        super(Jolyne,self).__init__()
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