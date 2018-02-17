
import os
from data import TripletLossDataset, EncoderDataset, NewDataset
from loss import BatchHardTripletLoss
import models
from models import ReidNetHardTrip
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau
from torch.autograd import Variable
from metric import CMC
from tools import Visualizer, parse_split_info, test
from config import opt
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def train(**kwargs):
    opt.parse(kwargs)

    env = Visualizer(opt.env)
    split_info = parse_split_info(opt.split_train_val)
    train_dataset = TripletLossDataset(train_root=opt.data_root, nb_select_class=18, nb_select_items=10,
                                       is_train=True, nb_time=1, specified_id_list=split_info['trainval'],
                                       resize_size=opt.resize_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1)
    valid_dataset = TripletLossDataset(train_root=opt.data_root, nb_select_class=2, nb_select_items=6,
                                       nb_time=1, is_train=False, specified_id_list=split_info['query'],
                                       resize_size=opt.resize_size)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1)
    cmc_valid_dataset = EncoderDataset(data_root=opt.data_root, specific_id_list=split_info['query'],
                                       resize_size=opt.resize_size)
    cmc_dataloader = DataLoader(dataset=cmc_valid_dataset, batch_size=12)
    reid_metric = CMC(dataloader=cmc_dataloader)

    reid_model = getattr(models, opt.model_name)(resnet_50=True)
    min_valid_loss = 1e6
    reid_model = nn.DataParallel(reid_model)
    if opt.use_gpu:
        reid_model.cuda()
    # optimizer = optim.SGD(reid_model.parameters(), lr=opt.lr, momentum=0.9)
    optimizer = optim.Adam(reid_model.parameters(), lr=opt.lr)
    criterion = BatchHardTripletLoss()

    epoch_size = opt.epoch_size
    scheduler = StepLR(optimizer=optimizer, step_size=10000 / len(train_dataset), gamma=0.2)
    top1_best = -1
    fw_top1_best = -1

    # Schedule learning rate

    def adjust_lr(epoch):
        t0 = 40
        lr = opt.lr if epoch <= t0 else \
            opt.lr * (0.001 ** ((epoch - t0) / opt.epoch_size - t0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    for epoch in range(epoch_size):

        reid_model.train()
        scheduler.step(epoch)
        for ii, (identity_list, image_data_list) in enumerate(train_dataloader):
            optimizer.zero_grad()
            image_data_list = image_data_list[0]
            identity_list = identity_list.t()
            dat_list = [identity_list, image_data_list]
            dat_list = [Variable(x) for x in dat_list]

            if opt.use_gpu:
                dat_list = [x.cuda() for x in dat_list]
            identity_list, image_data_list = dat_list
            feature_list = reid_model.forward(image_data_list)
            loss = criterion(feature_list, identity_list)
            loss.backward()
            optimizer.step()
            loss_log = loss.cpu().data[0]
            env.plot("loss", loss_log)
            print("epoch {} batch {} has loss {}".format(epoch, ii, loss_log))

        reid_model.eval()
        top1, top5, top10 = reid_metric.cmc(reid_model)
        env.plot("top1", top1)
        env.plot("top5", top5)
        env.plot("top10", top10)

        # test loss
        val_loss = val(reid_model, valid_dataloader, criterion, env)
        # scheduler.step(val_loss)

        if top1 > top1_best:
            top1_best = top1
            prefix = 'checkpoints/' + "ReidTriLoss_resnet50_top1" + '_' + str(top1) + "_"
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            torch.save(reid_model.state_dict(), name)


def fw_train():
    env = Visualizer(opt.env)
    split_info = parse_split_info(opt.split_train_val)
    valid_dataset = TripletLossDataset(train_root=opt.data_root, nb_select_class=2, nb_select_items=6,
                                       nb_time=1, is_train=False, specified_id_list=split_info['query'])
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1)
    cmc_valid_dataset = EncoderDataset(data_root=opt.data_root, resize_size=opt.resize_size,  specific_id_list=split_info['query'])
    cmc_dataloader = DataLoader(dataset=cmc_valid_dataset, batch_size=12)
    reid_metric = CMC(dataloader=cmc_dataloader)

    batch_hard_loss = BatchHardTripletLoss()

    def exp_lr_scheduler(optimizer, iters, iters_all=20000):
        lr = 0.01 * (0.2 ** (iters / iters_all))
        if iters % 20000 == 0:
            print 'LR is set to {}'.format(lr)
        for param in optimizer.param_groups:
            param['lr'] = lr
        return optimizer

    model = ReidNetHardTrip()
    min_valid_loss = 1e6
    model = nn.DataParallel(model)
    if opt.use_gpu:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    dataset = NewDataset(train_root="/home/xksj/Data/lp/re-identification/cuhk03-src/cuhk03_release/detected",
                         batch_size=110, max_per_id=10, is_train=True, resize_size=opt.resize_size, iter_sz=20000,
                         nb_time=1, specified_id_list=split_info['trainval'])
    train_loader = DataLoader(dataset=dataset, batch_size=1)


    for iteration, (labels, inputs) in enumerate(train_loader):
        optimizer = exp_lr_scheduler(optimizer, iteration)

        dat_list = [labels[0], inputs[0]]
        dat_list = [Variable(x) for x in dat_list]

        if opt.use_gpu:
            dat_list = [x.cuda() for x in dat_list]
        labels, inputs = dat_list
        model.train()
        features = model(inputs)
        batchloss = batch_hard_loss(features, labels)
        loss_log = batchloss.cpu().data[0]
        env.plot("loss", loss_log)
        optimizer.zero_grad()
        batchloss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            model.eval()
            rk_dict = test(model, opt.resize_size)
            top1 = rk_dict[1]
            top5 = rk_dict[5]
            env.plot("fw_top1", top1)
            env.plot("fw_top5", top5)

        if iteration % 200 == 0:
            val_loss = val(model, valid_dataloader, batch_hard_loss, env)
            top1, top5, top10 = reid_metric.cmc(model)
            env.plot("top1", top1)
            env.plot("top5", top5)
            env.plot("top10", top10)


def val(model, dataloader, criterion, env):
    model.eval()
    loss_log = 0.
    for ii, (identity_list, image_data_list) in enumerate(dataloader):
        image_data_list = image_data_list[0]
        identity_list = identity_list.t()
        dat_list = [identity_list, image_data_list]
        dat_list = [Variable(x) for x in dat_list]
        if opt.use_gpu:
            dat_list = [x.cuda() for x in dat_list]
        identity_list, image_data_list = dat_list
        feature_list = model.forward(image_data_list)
        loss = criterion(feature_list, identity_list)
        loss_log += loss.cpu().data[0]
    loss_log /= len(dataloader)
    env.plot("val_loss", loss_log)
    return loss_log

def eval(**kwargs):
    opt.parse(kwargs)

    env = Visualizer(opt.env)
    split_info = parse_split_info(opt.split_train_val)
    valid_dataset = TripletLossDataset(train_root=opt.data_root, nb_select_class=2, nb_select_items=6,
                                       nb_time=1, is_train=False, specified_id_list=split_info['query'],
                                       resize_size=opt.resize_size)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1)
    cmc_valid_dataset = EncoderDataset(data_root=opt.data_root, specific_id_list=split_info['query'],
                                       resize_size=opt.resize_size)
    cmc_dataloader = DataLoader(dataset=cmc_valid_dataset, batch_size=12)
    reid_metric = CMC(dataloader=cmc_dataloader)

    reid_model = getattr(models, opt.model_name)(resnet_50=True)
    reid_model.load(opt.best_module_path)
    min_valid_loss = 1e6
    reid_model = nn.DataParallel(reid_model)
    if opt.use_gpu:
        reid_model.cuda()

    epoch_size = opt.epoch_size
    top1_best = -1


    # Schedule learning rate

    top1, top5, top10 = reid_metric.cmc(reid_model)
    env.plot("top1", top1)
    env.plot("top5", top5)
    env.plot("top10", top10)



if __name__ == "__main__":
    train()
    # fuck()
    # fw_train()
