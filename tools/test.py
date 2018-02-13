import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models
from PIL import Image
import os
import copy
import numpy as np
import torch.nn.functional as F


class extract_model(nn.Module):
    def __init__(self):
        super(extract_model, self).__init__()
        model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1,
                                   model.layer2, model.layer3, model.layer4, nn.AvgPool2d(kernel_size=(5, 3)))

    def forward(self, inputs):
        inputs = self.model(inputs)
        outputs = inputs.view(-1, 512)
        outputs = F.normalize(outputs, p=2, dim=1)
        return outputs


def ComputeEuclid(feat_a, feat_b):
    feat_diff = feat_a - feat_b
    score = sum(feat_diff * feat_diff)
    return score


def GetRank(a1_feats, b1_feats, idx):
    feat_a1 = a1_feats[idx]
    feat_a1 = feat_a1.reshape(-1, )
    tmp_ranks = []
    for n in xrange(0, 100):
        feat_b1 = b1_feats[n]
        feat_b1 = feat_b1.reshape(-1, )
        score = ComputeEuclid(feat_a1, feat_b1)
        tmp_ranks.append((n, score))

    tmp_ranks = np.vstack(tmp_ranks)
    dist = tmp_ranks[:, 1]

    # rank
    idx_sort = np.argsort(tmp_ranks[:, 1])
    tmp_ranks = tmp_ranks[idx_sort, :]
    best_rank = -1
    for i in xrange(0, tmp_ranks.shape[0]):
        if (idx == tmp_ranks[i, 0]):
            best_rank = i + 1
            break

    return best_rank, dist


def test(model, resize_size):
    indir = 'query1/query1'
    imgtransforms = transforms.Compose([
        transforms.Scale(resize_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    a_feats = []
    b_feats = []
    files = os.listdir(indir)
    files.sort()

    for f in files:
        img_path = '{}/{}'.format(indir, f)
        img = Image.open(img_path).convert('RGB')
        # img = img.resize((resize_size[1], resize_size[0]))
        img = imgtransforms(img)
        img = torch.unsqueeze(img, dim=0)
        # img = img.resize_(1, 3, resize_size[0], resize_size[1])
        img = img.cuda()
        result = model(Variable(img))

        feat = result.data.cpu().numpy()
        tmp_feat = copy.deepcopy(feat)
        tmp_feat = tmp_feat.reshape(-1, )
        if f.split('_')[1] == '00':
            a_feats.append(tmp_feat)
        else:
            b_feats.append(tmp_feat)

    a_feats = np.vstack(a_feats)
    b_feats = np.vstack(b_feats)

    ranks = []
    dist = []
    for idx in xrange(0, 100):
        tmp_rank, dis = GetRank(a_feats, b_feats, idx)
        ranks.append(tmp_rank)
        dist.append(dis)

    for idx in xrange(0, 100):
        tmp_rank, dis = GetRank(b_feats, a_feats, idx)
        ranks.append(tmp_rank)
        dist.append(dis)

    ranks = np.vstack(ranks)
    dist = np.vstack(dist)

    sort = np.argsort(dist, 1)

    msort = np.zeros(sort.shape)
    for i in xrange(sort.shape[0]):
        for j in xrange(sort[0].shape[0]):
            msort[i][j] = np.where(sort[i] == j)[0]
    msort = msort + 1

    ranks = []

    for i in xrange(sort.shape[0]):
        ranks.append(np.where(sort[i] == i % 100)[0][0])
    ranks = np.vstack(ranks) + 1
    rank_acc_dict = {}
    for k in xrange(0, 2):
        if (k == 0):
            rank_thrd = 1
        else:
            rank_thrd = k * 5
        count = 0
        # for i in xrange(ranks.shape[0]/2,ranks.shape[0]):
        for i in xrange(0, ranks.shape[0]):
            if (ranks[i] <= rank_thrd):
                count += 1
        accuracy = float(count) / (ranks.shape[0])
        print 'Rank%d accuray=%f' % (rank_thrd, accuracy)
        rank_acc_dict[rank_thrd] = accuracy
    return rank_acc_dict


if __name__ == '__main__':
    model = extract_model()
    model.load_state_dict(
        torch.load('/home/feifei/workspace/experiment/pytorch/experiment/models/trip_res18/trip_res18_cuhk03_9999.pkl'))
    model = model.cuda()
    model.eval()
    test(model)
