# coding:utf8
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
from BasicModule import BasicModule


class ResNet(nn.Module):
    factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }
class ReidNetTripLoss(BasicModule):
    """
    Model for triplet loss
    """
    def __init__(self, id_num=751, net_name='resnet', pretrained=True, feature_dim=128):
        super(ReidNetTripLoss, self).__init__()
        self.id_num = id_num
        if net_name == 'resnet':
            # self.base = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=0)
            self.base = ResNet.factory[50](pretrained=pretrained)
            self.base_feature = self.base.fc.in_features
        else:
            raise NameError("net name is not predefined")
        self.feature = nn.Linear(self.base_feature, 1024)
        self.classifier = nn.Linear(1024, feature_dim)
        init.kaiming_normal(self.feature.weight, mode='fan_out')
        init.constant(self.feature.bias, 0)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)
        self.feat = nn.Sequential(self.feature,
                                  self.classifier)

    def avg_pool(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x

    def forward(self, *args):
        if len(args) == 3:
            a, b, c = args
            a_feature = self.feat(self.avg_pool(a))
            b_feature = self.feat(self.avg_pool(b))
            c_feature = self.feat(self.avg_pool(c))
            return a_feature, b_feature, c_feature
        else:
            feature = self.feat(self.avg_pool(args[0]))
            return feature
