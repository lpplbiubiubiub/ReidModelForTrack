#coding:utf8
import warnings


class DefaultConfig(object):
    model_name = "ReidNetHardTrip"
    split_train_val = "/home/xksj/Data/lp/re-identification/cuhk03-src/cuhk03_release/detected/splits.json"
    data_root = "/home/xksj/Data/lp/re-identification/cuhk03-src/cuhk03_release/detected"
    test_root = "/home/xksj/Data/lp/reid_train_data/corridor_test"

    env = "TripleLoss"
    resize_size = (100, 200)
    lr = 1e-4
    lr_decay = 0.1
    weight_decay = 1e-5
    epoch_size = 200
    decay_epoch_idx = 40
    use_gpu = True
    best_module_path = "checkpoints/ReidTriLoss_resnet50_top1_0.860759493671_0216_22:44:04.pth"

def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    # 更新配置参数
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            # 警告还是报错，取决于你个人的喜好
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    # 打印配置信息
    print('user config:')
    # self.__class__等价于调取类
    # __dict__ 获取属性
    # 在这里self指的是DefaultConfig的类属性
    for k, v in self.__class__.__dict__.iteritems():
        # 这里 __开头的是class自带的属性 要显示自定义属性
        if not k.startswith('__'):
            print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()