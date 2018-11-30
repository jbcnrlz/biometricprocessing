import torch.nn as nn

class GioGio(nn.Module):

    def calculateSize(self,dim,layer,inputSize):
        padding = layer.padding if (type(layer.padding) is not list) else layer.padding[dim]
        dilation = layer.dilation if (type(layer.dilation) is not list) else layer.dilation[dim]
        kernel_size = layer.kernel_size if (type(layer.kernel_size) is not list) else layer.kernel_size[dim]
        stride = layer.stride if (type(layer.stride) is not list) else layer.stride[dim]
        return int(((inputSize+(padding*2)-dilation*(kernel_size-1)-1) / stride) + 1)

    def __init__(self,classes,imageInput=(100,100)):
        self.imageInput = imageInput
        super(GioGio,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(384*5*5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 384*5*5)
        x = self.classifier(x)
        return x