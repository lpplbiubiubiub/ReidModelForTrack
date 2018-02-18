# coding:utf8
import torch as T
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from BasicModule import BasicModule
import torchvision


class ResNet(nn.Module):
    factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }


class ReidNetHardTrip(BasicModule):
    """
    Test base reid net
    """
    def __init__(self, pretrained=True, encoder=False, resnet_50=False):
        super(ReidNetHardTrip, self).__init__()
        if not resnet_50:
            self.base = ResNet.factory[18](pretrained=pretrained)
        else:
            self.base = ResNet.factory[50](pretrained=pretrained)
        self._resnet_50 = resnet_50
        self.nb_base_feature = self.base.fc.in_features
        self.feat = nn.Linear(self.nb_base_feature, 512)
        self.feat_bn = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
        self.embedding_feature = nn.Linear(512, 128)
        self.encoder = encoder
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        if self._resnet_50:
            # x = self.feat(x)
            pass
        return F.normalize(x)


class resnet_trip(nn.Module):
    def __init__(self):
        super(resnet_trip, self).__init__()
        model = ResNet.factory[18](pretrained=True)
        self.model = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1,
                                       model.layer2, model.layer3, model.layer4, nn.AvgPool2d(kernel_size=(5, 3)))

    def forward(self, inputs):
        inputs = self.model(inputs)
        outputs = inputs.view(-1, 512)
        outputs = F.normalize(outputs, p=2, dim=1)
        return outputs

