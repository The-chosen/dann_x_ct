import torch.nn as nn
import torch.nn.functional as F
from utils import ReverseLayerF
from options import options
import torchvision.models as models


# options
opt = options().opt
extractor = opt.extractor
classifier = opt.classifier
discriminator = opt.discriminator



class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        model = models.__dict__[extractor]()
        newmodel = nn.Sequential(*(list(model.children())[:-4]))
        self.extractor = newmodel

    def forward(self, x):
        x = self.extractor(x)
        print(x.size())
        x = x.view(-1, 128 * 32 * 32)
        return x

# class Extractor(nn.Module):
#     def __init__(self):
#         super(Extractor, self).__init__()
#         self.extractor = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),

#             nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )

#     def forward(self, x):
#         x = self.extractor(x)
#         x = x.view(-1, 3 * 256 * 256)
#         return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 32 * 32, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, x):
        x = self.classifier(x)
        return F.softmax(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=128 * 32 * 32, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return F.softmax(x)
