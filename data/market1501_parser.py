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

