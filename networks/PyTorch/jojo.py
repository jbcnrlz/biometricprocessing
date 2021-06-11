from PyTorchLayers.maxout_dynamic import *
from PyTorchLayers.octoconv import *
from PyTorchLayers.CorrelationImages import *
from networks.PyTorch.attentionModule import *
from networks.PyTorch.normActive import *
from scipy import stats
import math

def calculateMaxPoolingSize(inputsize,padding,dilatation,kernel,stride):
    return math.floor(((inputsize + (2 * padding) - dilatation * (kernel - 1) - 1) / stride)+1)

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

        '''
        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=8, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(896+in_channels, (896+in_channels)*2, kernel_size=1, stride=1),
            nn.BatchNorm2d((896+in_channels)*2),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=10, stride=5, padding=2)
        self.maxpoolb2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.maxInput = nn.MaxPool2d(kernel_size=25, stride=20)        
        self.features = nn.Sequential(
            nn.Dropout(),
            nn.Linear(((896+in_channels)*2)*4*4, 2048),

            nn.ReLU(inplace=True),
            MaxoutDynamic(int(2048 / 2), 2048),
            nn.Dropout(),
            nn.Linear(2048, 2048),
        )
        self.convNet = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=8, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=2, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        '''

        self.cv1 = nn.Conv2d(in_channels, 128, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(128)
        self.rl1 = nn.ReLU(inplace=True)
        self.cv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.rl2 = nn.ReLU(inplace=True)
        self.cv3 = nn.Conv2d(256, 512, kernel_size=4,stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.rl3 = nn.ReLU(inplace=True)
        self.cv4 = nn.Conv2d(512, 1024, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.rl4 = nn.ReLU(inplace=True)

        self.cv5 = nn.Conv2d(1920, 3840, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(3840)
        self.rl5 = nn.ReLU(inplace=True)

        self.features = nn.Sequential(
            nn.Dropout(),
            nn.Linear(34560, 2048),

            nn.ReLU(inplace=True),
            MaxoutDynamic(int(2048 / 2), 2048),
            nn.Dropout(),
            nn.Linear(2048, 2048),
        )


        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            MaxoutDynamic(int(2048 / 2), 2048),
            nn.Linear(2048, classes, bias=False)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=10, stride=7, padding=2)
        self.maxpoolb2 = nn.MaxPool2d(kernel_size=4, stride=3)
        self.maxpoolb3 = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        '''
        inputImage = self.maxInput(x)
        x = self.block0(x)
        ft1 = self.maxpool(x)
        x = self.block1(x)
        ft2 = self.maxpoolb2(x)
        x = self.block2(x)
        x = torch.cat((inputImage,ft1,ft2,x),dim=1)
        x = self.block3(x)
        x = x.view(x.size(0),-1)
        x = self.features(x)
        '''
        #x = self.convNet(x)
        x = self.cv1(x)
        ft1 = self.maxpool(x)
        x = self.bn1(x)
        x = self.rl1(x)
        x = self.cv2(x)
        ft2 = self.maxpoolb2(x)
        x = self.bn2(x)
        x = self.rl2(x)
        x = self.cv3(x)
        ft3 = self.maxpoolb3(x)
        x = self.bn3(x)
        x = self.rl3(x)
        x = self.cv4(x)
        x = self.bn4(x)
        x = self.rl4(x)
        x = torch.cat((ft1, ft2, ft3, x), dim=1)
        x = self.cv5(x)
        x = self.bn5(x)
        x = self.rl5(x)
        x = x.view(x.size(0),-1)
        x = self.features(x)
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
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6400, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes,bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return  self.softmax(x), x

def getBlock(in_channels,ks1,ks2,ks3):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=ks1, stride=int(ks1 / 2), padding=int(int(ks1 / 2)/2)),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=ks2, padding=int(ks2 / 2)),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(128, 256, kernel_size=ks3, padding=int(ks3 / 2)),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )

class GioGioModulateKernel(nn.Module):

    def calculateSize(self,dim,layer,inputSize):
        padding = layer.padding if (type(layer.padding) is not list) else layer.padding[dim]
        dilation = layer.dilation if (type(layer.dilation) is not list) else layer.dilation[dim]
        kernel_size = layer.kernel_size if (type(layer.kernel_size) is not list) else layer.kernel_size[dim]
        stride = layer.stride if (type(layer.stride) is not list) else layer.stride[dim]
        return int(((inputSize+(padding*2)-dilation*(kernel_size-1)-1) / stride) + 1)

    def __init__(self,classes,imageInput=(100,100),in_channels=4):
        self.imageInput = imageInput
        super(GioGioModulateKernel,self).__init__()
        self.features1 = getBlock(1, 8, 5, 3)
        self.features2 = getBlock(1, 6, 3, 2)
        self.features3 = getBlock(1, 6, 3, 2)
        self.features4 = getBlock(1, 3, 2, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25600, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, classes,bias=False)
        )

    def forward(self, x):
        x1 = x[:,0,:,:].reshape((-1,1,100,100))
        x1 = self.features1(x1)
        x2 = x[:,1,:,:].reshape((-1,1,100,100))
        x2 = self.features1(x2)
        x3 = x[:,2,:,:].reshape((-1,1,100,100))
        x3 = self.features1(x3)
        x4 = x[:,3,:,:].reshape((-1,1,100,100))
        x4 = self.features1(x4)
        x = torch.cat((x1,x2,x3,x4),axis=1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return  self.softmax(x), x

class GioGioModulateKernelInput(nn.Module):

    def calculateSize(self,dim,layer,inputSize):
        padding = layer.padding if (type(layer.padding) is not list) else layer.padding[dim]
        dilation = layer.dilation if (type(layer.dilation) is not list) else layer.dilation[dim]
        kernel_size = layer.kernel_size if (type(layer.kernel_size) is not list) else layer.kernel_size[dim]
        stride = layer.stride if (type(layer.stride) is not list) else layer.stride[dim]
        return int(((inputSize+(padding*2)-dilation*(kernel_size-1)-1) / stride) + 1)

    def __init__(self,classes,imageInput=(100,100)):
        self.imageInput = imageInput
        super(GioGioModulateKernelInput,self).__init__()

        self.input1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2,padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.input2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.input3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2,padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.input4 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.enFeat = FeatureEnhance(in_channels=64,out_channels=64)

        self.normInput = nn.Sequential(
            nn.LayerNorm((256,50,50)),
            nn.Conv2d(256,64,kernel_size=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6400, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, classes,bias=False)
        )

    def forward(self, x):
        x1 = x[:,0,:,:].reshape((-1,1,100,100))
        x1 = self.input1(x1)
        x2 = x[:,1,:,:].reshape((-1,1,100,100))
        x2 = self.input2(x2)
        x3 = x[:,2,:,:].reshape((-1,1,100,100))
        x3 = self.input3(x3)
        x4 = x[:,3,:,:].reshape((-1,1,100,100))
        x4 = self.input4(x4)
        x1, x2, x3, x4 = self.enFeat(x1,x2,x3,x4)
        x = torch.cat((x1,x2,x3,x4),axis=1)
        x = self.normInput(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return  self.softmax(x), x

class GioGioModulateKernelInputDepth(nn.Module):

    def __init__(self,classes,imageInput=(100,100)):
        self.imageInput = imageInput
        super(GioGioModulateKernelInputDepth,self).__init__()

        self.input1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2,padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.enFeat = FeatureEnhanceNoCross(in_channels=64,out_channels=64)

        self.features = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6400, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, classes,bias=False)
        )

    def forward(self, x):
        x = x[:,0,:,:].reshape((-1,1,100,100))
        x = self.input1(x)
        x = self.enFeat(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return  self.softmax(x), x



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    def __init__(self,inplanes,planes,stride=1,downsample=None,groups=1,norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes / 2)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width,width,kernel_size=3,stride=stride,groups=groups)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample =  nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3,stride=stride),
            norm_layer(planes),
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MaestroNetwork(nn.Module):
    def __init__(self,classes,imageInput=(100,100)):
        self.imageInput = imageInput
        super(MaestroNetwork,self).__init__()

        self.input1 = nn.Sequential(
            Bottleneck(1,64,2),
            Bottleneck(64, 128, 2),
            Bottleneck(128, 256, 2)
        )
        self.input2 = nn.Sequential(
            Bottleneck(1,64,2),
            Bottleneck(64, 128, 2),
            Bottleneck(128, 256, 2)
        )

        self.input3 = nn.Sequential(
            Bottleneck(1,64,2),
            Bottleneck(64, 128, 2),
            Bottleneck(128, 256, 2)
        )

        self.input4 = nn.Sequential(
            Bottleneck(1,64,2),
            Bottleneck(64, 128, 2),
            Bottleneck(128, 256, 2)
        )

        self.normLayer = nn.Sequential(
            nn.LayerNorm((1024,11,11)),
            nn.Conv2d(1024,512,stride=2,kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.feature = nn.Sequential(
            nn.Dropout(),
            nn.Linear(12800,1024),
            nn.ReLU(inplace=True)
        )

        self.softmax = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, classes,bias=False)
        )

    def forward(self, x):

        x1 = x[:,0,:,:].reshape((-1,1,100,100))
        x1 = self.input1(x1)
        x2 = x[:,1,:,:].reshape((-1,1,100,100))
        x2 = self.input2(x2)
        x3 = x[:,2,:,:].reshape((-1,1,100,100))
        x3 = self.input3(x3)
        x4 = x[:,3,:,:].reshape((-1,1,100,100))
        x4 = self.input4(x4)
        x = torch.cat((x1,x2,x3,x4),axis=1)
        x = self.normLayer(x)
        x = x.view(x.size(0),-1)
        x = self.feature(x)
        return self.softmax(x), x

class GioGioModulateKernelInputDepthDI(nn.Module):

    def __init__(self,classes,imageInput=(100,100)):
        self.imageInput = imageInput
        super(GioGioModulateKernelInputDepthDI,self).__init__()

        self.input1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2,padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.input2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.input3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2,padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.input4 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.input5 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2,padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.enFeat = FeatureEnhanceDepthDI(in_channels=64,out_channels=64)
        '''
        self.normInput = nn.Sequential(
            nn.LayerNorm((320,50,50)),
            nn.Conv2d(320,64,kernel_size=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        '''
        self.features = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=5, stride=2),
            nn.InstanceNorm2d(128),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6400, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout()
            #nn.Linear(4096, 4096),
        )

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, classes,bias=False)
        )

    def forward(self, x, xDepth):
        #ccalc = x.clone().cpu()
        x1 = x[:,0,:,:].reshape((-1,1,100,100))
        x2 = x[:,1,:,:].reshape((-1,1,100,100))
        x3 = x[:,2,:,:].reshape((-1,1,100,100))
        x4 = x[:,3,:,:].reshape((-1,1,100,100))
        #print('Inicial')
        #print(stats.pearsonr(ccalc[0,0,:,:].flatten(),ccalc[0,1,:,:].flatten()))
        #print(stats.pearsonr(ccalc[0,0,:,:].flatten(),ccalc[0,2,:,:].flatten()))
        #print(stats.pearsonr(ccalc[0,0,:,:].flatten(),ccalc[0,3,:,:].flatten()))
        x1 = self.input1(x1)
        x2 = self.input2(x2)
        x3 = self.input3(x3)
        x4 = self.input4(x4)
        x5 = self.input5(xDepth[:,0,:,:].reshape((-1,1,100,100)))
        x1, x2, x3, x4,x5 = self.enFeat(x1,x2,x3,x4,x5)
        #print('Att Maps')
        #print(stats.pearsonr(x1[0,:,:,:].clone().cpu().flatten(),x2[0,:,:,:].clone().cpu().flatten()))
        #print(stats.pearsonr(x1[0,:,:,:].clone().cpu().flatten(), x3[0,:,:,:].clone().cpu().flatten()))
        #print(stats.pearsonr(x1[0,:,:,:].clone().cpu().flatten(), x4[0,:,:,:].clone().cpu().flatten()))
        x = torch.cat((x1,x2,x3,x4,x5),axis=1)
        #x = self.normInput(x)
        #print('After Norm')
        #xNormed = x.clone().cpu()
        #for idxChan in range(1,xNormed.shape[1]):
        #    print(stats.pearsonr(xNormed[0,0,:,:].flatten(),xNormed[0,idxChan,:,:].flatten()))
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return  self.softmax(x), x