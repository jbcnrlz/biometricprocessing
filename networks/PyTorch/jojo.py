import torch.nn as nn


class GioGio(nn.Module):

    def __init__(self,classes):
        super(GioGio,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=8, stride=8, padding=2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*13*13, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64*13*13)
        x = self.classifier(x)
        return x