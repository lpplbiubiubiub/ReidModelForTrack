# coding:utf8
import torch as t
import torch.nn as nn
from torch.nn import init

import time

class BasicModule(t.nn.Module):
    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, model_path):
        """
        load model according name
        """
        model_dict = self.state_dict()
        pretrained_dict = t.load(model_path)
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        self.load_state_dict(model_dict)

    def save(self, name=None):
        """
        save model
        """
        if name is None:
           prefix = 'checkpoints/' + self.model_name + '_'
           name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)