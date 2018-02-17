from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from itertools import combinations
import random


from scipy.spatial.distance import cdist


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, feature, identity):
        assert feature.size(0) == identity.size(0), "feature and id should have same nb"
        n = feature.size(0)
        # P: class K: item in one class
        # n should equals PK
        # use numpy, scipy for unique, pdist opr
        # assuming feature and identity on GPU
        # should get two mask with elem is (x1, y2) and (x2, y2)
        # so we can get hardert positive and hardest negative
        feature_numpy = feature.cpu().data.numpy()
        identity_numpy = identity.cpu().data.numpy().flatten()
        feature_dis_mat = cdist(feature_numpy, feature_numpy)
        pos_hardest = []
        neg_hardest = []
        for i in range(n):
            identity_tmp = np.arange(n)[identity_numpy[i] == identity_numpy, np.newaxis]
            unidentity_tmp = np.arange(n)[identity_numpy[i] != identity_numpy, np.newaxis]

            feature_dis_mat_pos = feature_dis_mat[i].copy()
            feature_dis_mat_pos[unidentity_tmp] = 0.
            hardest_pos_absolute_pos = (i, feature_dis_mat_pos.argmax())
            pos_hardest.append(hardest_pos_absolute_pos)
            feature_dis_mat_neg = feature_dis_mat[i].copy()
            feature_dis_mat_neg[identity_tmp] = np.inf
            hardest_neg_absolute_pos = (i, feature_dis_mat_neg.argmin())
            neg_hardest.append(hardest_neg_absolute_pos)

        loss = 0.
        for pos, neg in zip(pos_hardest, neg_hardest):
            pos_hardest_dis = F.pairwise_distance(torch.unsqueeze(feature[pos[0]], 0), torch.unsqueeze(feature[pos[1]], 0))
            neg_hardest_dis = F.pairwise_distance(torch.unsqueeze(feature[neg[0]], 0), torch.unsqueeze(feature[neg[1]], 0))
            loss += F.relu(self.margin + pos_hardest_dis - neg_hardest_dis)
        loss = loss / n
        return loss[0]

class BatchHardTripletBlendLoss(nn.Module):
    """
    Blend triplet batch hard model with normal triplet loss
    """
    def __init__(self, margin=0.2, begin_downscale=20, down_size=10, down_scale=0.2):
        """
        :param margin: pair between hard should exceed positive margin
        :param begin_downscale: epoch when downscale
        :param down_size: after begin downscale, after per scale, down size
        :param down_scale:
        """
        super(BatchHardTripletBlendLoss, self).__init__()
        self.margin = margin
        self.start_downscale = begin_downscale
        self.down_size = down_size
        self.down_scale = down_scale
        self.epoch = 0

    def forward(self, feature, identity, epoch):
        assert feature.size(0) == identity.size(0), "feature and id should have same nb"
        n = feature.size(0)
        batch_hard_ratio = 0.
        if epoch > self.start_downscale:
            batch_hard_ratio = self.down_scale * ((epoch - self.start_downscale) // self.down_size + 1)
            if batch_hard_ratio > 1.:
                batch_hard_ratio = 1.
        feature_numpy = feature.cpu().data.numpy()
        identity_numpy = identity.cpu().data.numpy().flatten()
        feature_dis_mat = cdist(feature_numpy, feature_numpy)
        pos_hardest = []
        neg_hardest = []
        pos_normal = []
        neg_normal = []
        for i in range(n):
            identity_tmp = np.arange(n)[identity_numpy[i] == identity_numpy, np.newaxis]
            unidentity_tmp = np.arange(n)[identity_numpy[i] != identity_numpy, np.newaxis]
            if i < int(n * batch_hard_ratio) and batch_hard_ratio > 0: # need to have 2
                # normal batch
                identity_list = identity_tmp.tolist()
                unidentity_list = unidentity_tmp.tolist()
                identity_list.remove([i])
                # unidentity_list.remove([i])
                # sample balance
                unidentity_list = random.sample(unidentity_list, len(identity_list))
                pos_pair = None
                neg_pair = None
                for pos_idx in identity_list:
                    pos_idx = pos_idx[0]
                    if i < pos_idx:
                        pos_pair = (i, pos_idx)
                    else:
                        pos_pair = (pos_idx, i)
                    pos_normal.append(pos_pair)
                for neg_idx in unidentity_list:
                    neg_idx = neg_idx[0]
                    if i < neg_idx:
                        neg_pair = (i, neg_idx)
                    else:
                        neg_pair = (neg_idx, i)
                    neg_normal.append(neg_pair)
            else:
                feature_dis_mat_pos = feature_dis_mat[i].copy()
                feature_dis_mat_pos[unidentity_tmp] = 0.
                if i < feature_dis_mat_pos.argmax():
                    hardest_pos_absolute_pos = (i, feature_dis_mat_pos.argmax())
                else:
                    hardest_pos_absolute_pos = (feature_dis_mat_pos.argmax(), i)
                pos_hardest.append(hardest_pos_absolute_pos)
                feature_dis_mat_neg = feature_dis_mat[i].copy()
                feature_dis_mat_neg[identity_tmp] = np.inf
                if i < feature_dis_mat_neg.argmin():
                    hardest_neg_absolute_pos = (i, feature_dis_mat_neg.argmin())
                else:
                    hardest_neg_absolute_pos = (feature_dis_mat_neg.argmin(), i)
                neg_hardest.append(hardest_neg_absolute_pos)

        loss = 0.
        for pos, neg in zip(pos_hardest, neg_hardest):
            pos_hardest_dis = F.pairwise_distance(torch.unsqueeze(feature[pos[0]], 0),
                                                  torch.unsqueeze(feature[pos[1]], 0))
            neg_hardest_dis = F.pairwise_distance(torch.unsqueeze(feature[neg[0]], 0),
                                                  torch.unsqueeze(feature[neg[1]], 0))
            loss += F.relu(self.margin + pos_hardest_dis - neg_hardest_dis)
        for pos, neg in zip(pos_normal, neg_normal):
            pos_hardest_dis = F.pairwise_distance(torch.unsqueeze(feature[pos[0]], 0),
                                                  torch.unsqueeze(feature[pos[1]], 0))
            neg_hardest_dis = F.pairwise_distance(torch.unsqueeze(feature[neg[0]], 0),
                                                  torch.unsqueeze(feature[neg[1]], 0))

            loss += F.relu(self.margin + pos_hardest_dis - neg_hardest_dis)
        loss = loss / (len(pos_hardest) + len(pos_normal))
        return loss[0]

if __name__ == "__main__":
    feature = Variable(torch.rand(72, 128))
    identity = Variable((torch.rand(72) * 10).int() % 4)
    loss = BatchHardTripletBlendLoss(0.3)
    loss.forward(feature, identity)
