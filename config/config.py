#coding:utf8
import warnings


class DefaultConfig(object):
    env = 'main'  # visdom 环境
    model = 'ReidNetV2'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    triplet_flag = False
    data_per_time = 1  # 每次获取的数据量

    basenet = 'vgg16'
    train_data_root = '/home/xksj/Data/lp/re-identification/cuhk03_release/detected'  # 训练集存放路径
    valid_data_root = '/home/xksj/Data/lp/re-identification/cuhk03_release/test'
    finetune_data_root = '/home/xksj/Data/lp/reid_finetune/sample_train_seq'
    finetune_valid_root = '/home/xksj/Data/lp/reid_finetune/sample_valid_seq'
    query_data_root = '/home/xksj/Data/lp/re-identification/cuhk03_release/test'
    test_data_root = '/home/xksj/Data/lp/re-identification/cuhk03_release/test'
    load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 72 # batch size
    val_batch_size = 4
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 140
    lr = 0.01 # initial learning rate (if softmax 0.01)
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4  # 正则
    momentum = 0.9
    step_size = 70

    # dataset相关
    resize_sz = (256, 256)
    crop_sz = (224, 224)
    # resize_sz = (313, 313)
    # crop_sz = (299, 299)

    # 模型相关
    id_num = 1300

    # 预训练模型
    pretrain_resnet_weight_root = "/home/xksj/.torch/models/resnet50-19c8e357.pth"
    pretrain_alexnet_weight_root = "/home/xksj/.torch/models/alexnet-owt-4df8aa71.pth"
    weight_pre_name = "base" # 骨干网络前缀名 用于载入预训练模型



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