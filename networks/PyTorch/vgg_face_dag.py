import torch
import torch.nn as nn
from helper.functions import shortenNetwork

class vgg_smaller(nn.Module):

    def __init__(self,fullVgg,numClasses=None,features=4096,bufferCenters=False):
        super(vgg_smaller, self).__init__()
        self.sizeReduction = nn.Sequential(
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        )
        self.convolutional = nn.Sequential(*list(fullVgg.children())[:-7])
        self.fullyConnected = nn.Sequential(*list(fullVgg.children())[-7:-3])
        self.softmax = None

        if bufferCenters:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.register_buffer('centers', (torch.rand(numClasses, features).to(torch.device(device)) - 0.5) * 2)

        if numClasses is not None:
            self.softmax = nn.Linear(in_features=features, out_features=numClasses,bias=False)

    def forward(self, x0):
        x0 = self.convolutional(x0)
        x0 = x0.view(x0.size(0), -1)
        x0 = self.fullyConnected(x0)
        if self.softmax is not None:
            return x0, self.softmax(x0)
        else:
            return x0, None

class vgg_face_dag(nn.Module):

    def __init__(self):
        super(vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x0 = self.conv1_1(x0)
        x0 = self.relu1_1(x0)
        x0 = self.conv1_2(x0)
        x0 = self.relu1_2(x0)
        x0 = self.pool1(x0)
        x0 = self.conv2_1(x0)
        x0 = self.relu2_1(x0)
        x0 = self.conv2_2(x0)
        x0 = self.relu2_2(x0)
        x0 = self.pool2(x0)
        x0 = self.conv3_1(x0)
        x0 = self.relu3_1(x0)
        x0 = self.conv3_2(x0)
        x0 = self.relu3_2(x0)
        x0 = self.conv3_3(x0)
        x0 = self.relu3_3(x0)
        x0 = self.pool3(x0)
        x0 = self.conv4_1(x0)
        x0 = self.relu4_1(x0)
        x0 = self.conv4_2(x0)
        x0 = self.relu4_2(x0)
        x0 = self.conv4_3(x0)
        x0 = self.relu4_3(x0)
        x0 = self.pool4(x0)
        x0 = self.conv5_1(x0)
        x0 = self.relu5_1(x0)
        x0 = self.conv5_2(x0)
        x0 = self.relu5_2(x0)
        x0 = self.conv5_3(x0)
        x0 = self.relu5_3(x0)
        x0 = self.pool5(x0)
        x0 = x0.view(x0.size(0), -1)
        x0 = self.fc6(x0)
        x0 = self.relu6(x0)
        x0 = self.dropout6(x0)
        x0 = self.fc7(x0)
        x0 = self.relu7(x0)
        x0 = self.dropout7(x0)
        x0 = self.fc8(x0)
        return x0

def vgg_face_dag_load(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = vgg_face_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

def finetuned_vgg_face_dag_load(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = vgg_face_dag()
    model = vgg_smaller(model)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.fullyConnected = nn.Sequential(
            nn.Linear(in_features=state_dict['state_dict']['fullyConnected.0.weight'].shape[1], out_features=4096, bias=True),
            *list(model.fullyConnected.children())[1:],
            nn.Linear(in_features=2622, out_features=state_dict['state_dict']['fullyConnected.7.weight'].shape[0])
        )
        #model.fullyConnected.add_module('7', nn.Linear(in_features=2622, out_features=state_dict['state_dict']['fullyConnected.7.weight'].shape[0]))
        model.load_state_dict(state_dict['state_dict'])
        model.fullyConnected = nn.Sequential(*list(model.fullyConnected.children())[:-1])

    return model


def small_vgg_face_dag_load(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = vgg_face_dag()
    model = vgg_smaller(model)
    shortenedList = shortenNetwork(
        list(model.convolutional.children()),
        list(range(17))
    )
    model.convolutional = nn.Sequential(*shortenedList)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.fullyConnected = nn.Sequential(
            nn.Linear(in_features=state_dict['state_dict']['fullyConnected.0.weight'].shape[1], out_features=4096, bias=True),
            *list(model.fullyConnected.children())[1:],
            nn.Linear(in_features=2622, out_features=state_dict['state_dict']['fullyConnected.7.weight'].shape[0])
        )
        #model.fullyConnected.add_module('7', nn.Linear(in_features=2622, out_features=state_dict['state_dict']['fullyConnected.7.weight'].shape[0]))
        model.load_state_dict(state_dict['state_dict'])
        model.fullyConnected = nn.Sequential(*list(model.fullyConnected.children())[:-1])

    return model

def smaller_vgg_face_dag_load(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = vgg_face_dag()
    model = vgg_smaller(model)
    shortenedList = shortenNetwork(
        list(model.convolutional.children()),
        [0, 1, 4, 5, 6, 9, 10, 11, 16]
    )
    model.convolutional = nn.Sequential(*shortenedList)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.fullyConnected.add_module('7', nn.Linear(in_features=2622, out_features=state_dict['state_dict']['fullyConnected.7.weight'].shape[0]))
        model.load_state_dict(state_dict['state_dict'])
        model.fullyConnected = nn.Sequential(*list(model.fullyConnected.children())[:-1])

    return model

def medium_vgg_face_dag_load(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = vgg_face_dag()
    model = vgg_smaller(model)
    shortenedList = shortenNetwork(
        list(model.convolutional.children()),
        [0, 1, 4, 5, 6, 9, 10, 11, 16, 17, 18, 23],True
    )
    model.convolutional = nn.Sequential(*shortenedList)

    if weights_path:
        state_dict = torch.load(weights_path)

        model.fullyConnected = nn.Sequential(
            nn.Linear(in_features=state_dict['state_dict']['fullyConnected.0.weight'].shape[1], out_features=4096, bias=True),
            *list(model.fullyConnected.children())[1:]
        )

        model.softmax = nn.Linear(in_features=state_dict['state_dict']['softmax.weight'].shape[1], out_features=state_dict['state_dict']['softmax.weight'].shape[0],bias=False)
        model.load_state_dict(state_dict['state_dict'])

    return model

def centerloss_vgg_face_dag_load(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    state_dict = torch.load(weights_path)
    model = vgg_face_dag()
    model = vgg_smaller(model,state_dict['state_dict']['softmax.weight'].shape[0],bufferCenters=True)
    shortenedList = shortenNetwork(
        list(model.convolutional.children()),
        [0, 1, 4, 5, 6, 9, 10, 11, 16, 17, 18, 23], True
    )
    model.convolutional = nn.Sequential(*shortenedList)
    model.fullyConnected = nn.Sequential(
        nn.Linear(in_features=18432, out_features=4096),
        *list(model.fullyConnected.children())[1:]
    )

    model.fullyConnected = nn.Sequential(
        nn.Linear(in_features=state_dict['state_dict']['fullyConnected.0.weight'].shape[1], out_features=4096, bias=True),
        *list(model.fullyConnected.children())[1:]
    )
    #model.softmax = nn.Linear(in_features=2622, out_features=state_dict['state_dict']['softmax.weight'].shape[0])
    model.load_state_dict(state_dict['state_dict'])

    return model