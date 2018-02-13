from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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


if __name__ == "__main__":
    feature = Variable(torch.rand(72, 128))
    identity = Variable((torch.rand(72) * 10).int() % 4)
    loss = BatchHardTripletLoss(0.3)
    loss.forward(feature, identity)
