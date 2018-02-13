# coding:utf8
import os
from PIL import Image
import torch
from torch.utils import data
from torch import tensor
from torch import FloatTensor, IntTensor
import numpy as np
from torchvision import transforms as T
import random
import sys
import time

IMG_EXT = "jpg"


def parse_market_data(dataset_root="/home/xksj/Data/lp/re-identification/Market-1501-v15.09.15/bounding_box_train",
                      specific_id_list = None, embedding= None):
    """
    parse dataset path as a dict
    the key is identify and the value is serial name
    like 0002_c1s1_000421_03.jpg
    0002 is the identity and c1s1_000421_03 is serial name
    train_id_list should have value like "0011"
    """
    filename_list = []
    dataset_dict = {}
    embeding_dict = {}
    for parent, pathnames, filenames in os.walk(dataset_root):
        for filename in filenames:
            if filename.split(".")[-1] == IMG_EXT:
                identity = filename.split("_")[0]
                if int(identity) in specific_id_list or specific_id_list is None: # split junk out
                    list.append(filename_list, filename.split(".")[0])
    for filename in filename_list:
        identity = int(filename.split("_")[0])
        if not dict.has_key(dataset_dict, identity):
            dataset_dict[identity] = []
        list.append(dataset_dict[identity], filename)
    for idx, val in enumerate(dataset_dict.keys()):
        embeding_dict[int(val)] = idx
    if embedding is not None:
        assert len(embedding.keys()) == len(embeding_dict.keys()), "embedding keys' count should be same"
        embeding_dict = embedding
    return filename_list, dataset_dict, embeding_dict


class TripletLossDataset(data.Dataset):
    """
    A dataset class for triplet loss
    NOTICE:
        there is no need for embedding, because triplet loss have no embedding input
    """
    def __init__(self, train_root="", nb_select_class=13, nb_select_items=10, is_train=True, resize_size=(128, 256), nb_time=1, specified_id_list = None):
        super(TripletLossDataset, self).__init__()
        assert os.path.exists(train_root), "data path is not exist"
        if type(specified_id_list) is list:
            assert len(specified_id_list) > 0, "specific id list should not be none"
        self._data_root = train_root
        self._im_list, self._id_seq_dict, self._embedding = parse_market_data(train_root, specific_id_list=specified_id_list)

        self._nb_id = len(self._id_seq_dict)
        assert nb_select_class < self._nb_id, "nb of class exceed nb of class in this dataset"
        self._id_list = map(lambda x: int(x), self._id_seq_dict.keys())
        random.shuffle(self._id_list)



        self._nb_id_per_batch = nb_select_class
        self._nb_data_per_class = nb_select_items
        self._nb_pair = len(self._id_list) / self._nb_id_per_batch

        self._nb_time = nb_time
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        # ATTENTION: SCALE RECIEVE PARAM: width height
        if is_train:
            self._transform = T.Compose([
                T.Scale(resize_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self._transform = T.Compose([
                T.Scale(resize_size),
                T.ToTensor(),
                normalize
            ])
        random.seed(time.time())

    def __getitem__(self, index):
        final_id_list = []
        final_img_list = []
        # random select ids
        batch_id_list = random.sample(self._id_list, self._nb_id_per_batch)
        for identity in batch_id_list:
            src_same_identity_file_list = self._id_seq_dict[identity]
            nb_data_per_class = len(src_same_identity_file_list)
            same_identity_file_list = []
            if nb_data_per_class >= self._nb_data_per_class:
                same_identity_file_list = random.sample(src_same_identity_file_list, self._nb_data_per_class)
            else:
                [same_identity_file_list.append(random.choice(src_same_identity_file_list)) for i in range(self._nb_data_per_class)]
            # for id
            [final_id_list.append(self._embedding[identity]) for i in range(self._nb_data_per_class)]
            # for img data
            [final_img_list.append(self._transform(Image.open(os.path.join(self._data_root, img + ".jpg")).convert('RGB'))) for img in same_identity_file_list]
        return IntTensor(final_id_list), torch.stack(final_img_list, dim=0)

    def __len__(self):
        return len(self._id_list) / (self._nb_id_per_batch)

class NewDataset(data.Dataset):
    """
    A dataset class for triplet loss
    NOTICE:
        there is no need for embedding, because triplet loss have no embedding input
    """
    def __init__(self, train_root="", batch_size=100, max_per_id=10, is_train=True, resize_size=(200, 100), iter_sz = 20000,  nb_time=1, specified_id_list = None):
        super(NewDataset, self).__init__()
        assert os.path.exists(train_root), "data path is not exist"
        if type(specified_id_list) is list:
            assert len(specified_id_list) > 0, "specific id list should not be none"
        self._data_root = train_root
        self._im_list, self._id_seq_dict, self._embedding = parse_market_data(train_root, specific_id_list=specified_id_list)

        self._nb_id = len(self._id_seq_dict)
        self._id_list = map(lambda x: int(x), self._id_seq_dict.keys())
        self.all_id = self._id_seq_dict.keys()
        random.shuffle(self.all_id)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        self._batch_sz = batch_size
        self._max_item_per_id = max_per_id
        self.id_order = 0
        self._iter_sz = iter_sz

        self.h = resize_size[0]
        self.w = resize_size[1]

        if is_train:
            self._transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self._transform = T.Compose([
                T.ToTensor(),
                normalize
            ])
        random.seed(time.time())

    def get_batch_list(self):
        count = 0
        while count < self._batch_sz:
            if self.id_order >= len(self.all_id):
                random.shuffle(self.all_id)
                self.id_order = 0
            identity = self.all_id[self.id_order]
            if len(self._id_seq_dict[identity]) >= self._max_item_per_id:
                same_id_file = random.sample(self._id_seq_dict[identity], self._max_item_per_id)
            else:
                same_id_file = self._id_seq_dict[identity][:]
            for file_name in same_id_file:
                if count < self._batch_sz:
                    count += 1
                    yield(file_name, identity)
                else:
                    return
            self.id_order += 1

    def __getitem__(self, index):
        batch_list = self.get_batch_list()
        final_id_list = []
        final_img_list = []
        for filename, identity in batch_list:
            # for id
            final_id_list.append(self._embedding[identity])
            # for img data
            image = Image.open(os.path.join(self._data_root, filename + ".jpg")).convert('RGB').resize((self.w, self.h))
            final_img_list.append(self._transform(image))
        return IntTensor(final_id_list), torch.stack(final_img_list, dim=0)

    def __len__(self):
        return self._iter_sz

class EncoderDataset(data.Dataset):
    def __init__(self, data_root="/home/xksj/Data/lp/re-identification/Market-1501-v15.09.15/bounding_box_train",
                 resize_size=(200, 100), specific_id_list = None
                 ):
        file_list = []
        for parent, pathnames, filenames in os.walk(data_root):
            for filename in filenames:
                if filename.split(".")[-1] == IMG_EXT:
                    identity = filename.split("_")[0]
                    if specific_id_list is None:
                        list.append(file_list, filename.split(".")[0])
                    elif int(identity) in specific_id_list:  # split junk out
                        list.append(file_list, filename.split(".")[0])
        self._data_root = data_root
        self._id_list = map(lambda x: int(x.split('_')[0]), file_list)
        self._cam_id_list = map(lambda x: int(x.split('_')[1]), file_list)
        self._img_list = map(lambda x: os.path.join(data_root, x + "." + IMG_EXT), file_list)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        self._transform = T.Compose([
            T.Scale(resize_size),
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        img_path = self._img_list[index]
        identity = self._id_list[index]
        cam_id = self._cam_id_list[index]
        assert os.path.isfile(img_path), "image dose not exist" + " and img path is " + img_path
        image_data = self._transform(Image.open(img_path).convert('RGB'))
        return image_data, identity, cam_id

    def __len__(self):
        return len(self._img_list)



if __name__ == "__main__":
    dataset = TripletLossDataset(train_root="/home/xksj/Data/lp/reid_train_data/corridor_train")
    for i in range(len(dataset)):
        id_list, img_list = dataset[i]


