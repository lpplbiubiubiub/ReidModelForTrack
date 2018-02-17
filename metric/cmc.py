# from torch.autograd import Variable
import numpy as np
from scipy.spatial.distance import cdist, norm
import random
import time
import sys
from torch.autograd import Variable


class CMC(object):
    def __init__(self, dataloader=None):
        self._valid_dataloader = dataloader
        random.seed(time.time())

    def cmc(self, model):
        """
        1. How to compute CMC?

        We use one camera view as the probe set, and the other as the gallery set.
        For the gallery, we randomly sample one image for each identity.
        For the probe, we use all the images, getting the CMC curve for each of them, and then average over them.
        This evaluation process is repeated for 100 times and the mean value is reported as the final result.
        See the Matlab function for more details.
        :param model:
        :return:
        """
        model.eval()
        final_feature = None
        final_id_list = []
        final_cam_id_list = []

        for ii, (image_data_list, id_data_list, cam_id_list) in enumerate(self._valid_dataloader):
            final_id_list.extend(id_data_list.numpy().flatten().tolist())
            final_cam_id_list.extend(cam_id_list.numpy().flatten().tolist())
            dat_list = [image_data_list]
            dat_list = [Variable(x) for x in dat_list]
            dat_list = [x.cuda() for x in dat_list]
            image_data_list = dat_list[0]
            feature_list = model.forward(image_data_list)
            feature_list_arr = feature_list.cpu().data.numpy()
            if final_feature is None:
                final_feature = feature_list_arr
            else:
                final_feature = np.vstack([final_feature, feature_list_arr])
            norm_arr = norm(final_feature, ord=2, axis=-1)[:, np.newaxis]
            final_feature /= norm_arr

        final_id_list = np.array(final_id_list)
        final_cam_id_list = np.array(final_cam_id_list)
        unique_id, unique_idx = np.unique(final_id_list, return_inverse=True)
        nb_id = unique_id.shape[0]
        nb_data = final_id_list.shape[0]
        count = 0
        probe_view = random.randint(0, 1)
        gallery_view = 1 - probe_view
        probe_view_list = final_cam_id_list == probe_view
        gallery_view_list = final_cam_id_list == gallery_view

        range_arr = np.arange(nb_data)
        gallery_view_idx_list = []
        probe_view_idx_list = range_arr[probe_view_list]
        for identity in unique_id:
            same_id_list = identity == final_id_list
            same_id_list *= gallery_view_list
            sample_idx = random.choice(range_arr[same_id_list])
            gallery_view_idx_list.append(sample_idx)

        gallery_view_idx_list = np.array(gallery_view_idx_list)

        probe_identity_list = final_id_list[probe_view_idx_list]
        gallery_identity_list = final_id_list[gallery_view_idx_list]
        probe_feature_list = final_feature[probe_view_idx_list]
        gallery_feature_list = final_feature[gallery_view_idx_list]
        mat_dist = cdist(probe_feature_list, gallery_feature_list)
        sort_mat = np.argsort(mat_dist, axis=-1)
        # compute cmc
        # top1 top5 top10
        top1 = 0
        top5 = 0
        top10 = 0
        for i, sort_res in enumerate(sort_mat):
            match_res = probe_identity_list[i] == gallery_identity_list[sort_res]
            top1 += match_res[0]
            top5 += 1 if np.sum(match_res[0:5]) > 0 else 0
            top10 += 1 if  np.sum(match_res[:10]) > 0 else 0
        # compute top10
        return top1 / (0. + mat_dist.shape[0]), top5 / (0. + mat_dist.shape[0]), top10 / (0. + mat_dist.shape[0])

if __name__ == "__main__":
    pass