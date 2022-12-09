import os
import sys
import time
import torch
from copy import deepcopy

import torch.nn as nn
import torch.nn.init as init
import torch.distributed as dist
import logging
import os
from collections import OrderedDict
import math
import torch.nn.functional as F

import re
from einops import rearrange

# DALI imports

from nvidia.dali.plugin.pytorch import DALIGenericIterator

import os
import sys
import time
import torch
import pickle
import numpy as np
import nvidia.dali.ops as ops

from torchvision import datasets
from sklearn.utils import shuffle
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms

_logger = logging.getLogger(__name__)

def monarch_weight_to_dense_weight(w1_bfly, w2_bfly):
    """
    Argumments:
        w1_bfly: (nblocks, out / nblocks, in / blocks)
        w2_bfly: (nblocks, out / nblocks, in / blocks)
    Return:
        dense_weight: (out / in)
    """
    # batch, n = x.shape
    device = w1_bfly.device

    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape

    w1_dense = torch.block_diag(*torch.unbind(w1_bfly, dim=0))
    w2_dense = torch.block_diag(*torch.unbind(w2_bfly, dim=0))

    P_1 = rearrange(torch.eye(k*q), 'b (r l) -> b (l r)', l=l).to(device)
    P_2_T = rearrange(torch.eye(l*s), 'b (l s) -> b (s l)', l=4).to(device)
    P_1_T = P_1.T 
    P_2 = P_2_T.T 

    L = w2_dense
    R = w1_dense

    M = P_2 @ L @ P_1_T @ R 
    return M

def monarch_to_dense_mlp_NC(state_dict, new_state_dict):
    # from src.ops.blockdiag_multiply import blockdiag_weight_to_dense_weight
    blkdiag1_names = sorted({name for name in state_dict
                             if re.match('layers.(\d+).NC.(\d+).mlp.fc(1|2).blkdiag1',
                                          name)})
    blkdiag2_names = sorted({name for name in state_dict
                             if re.match('layers.(\d+).NC.(\d+).mlp.fc(1|2).blkdiag2',
                                          name)})
    _logger.info('Processing S2D: Normal Cell Monarch to Mlp')
    # print(blkdiag1_names)
    # print(blkdiag2_names)
    for blkdiag1_name in blkdiag1_names:
        try:
            blkdiag2_name = blkdiag1_name.replace('blkdiag1', 'blkdiag2')
            new_name = blkdiag1_name.replace('blkdiag1', 'weight')
            new_state_dict[new_name] = monarch_weight_to_dense_weight(state_dict[blkdiag1_name], state_dict[blkdiag2_name])
        except:
            _logger.error("blkdiag and mlp name or device do not match")
        # blkdiag2_name = blkdiag1_name.replace('blkdiag1', 'blkdiag2')
        # new_name = blkdiag1_name.replace('blkdiag1', 'weight')
        # new_state_dict[new_name] = monarch_weight_to_dense_weight(state_dict[blkdiag1_name], state_dict[blkdiag2_name])
    return new_state_dict


# copied from https://github.com/tanglang96/DataLoaders_DALI/
class DALIDataloader(DALIGenericIterator):
    def __init__(self, pipeline, size, batch_size, output_map=["data", "label"], auto_reset=True, onehot_label=False):
        self.size = size
        self.batch_size = batch_size
        self.onehot_label = onehot_label
        self.output_map = output_map
        super().__init__(pipelines=pipeline, size=size, auto_reset=auto_reset, output_map=output_map)

    # https://github.com/tanglang96/DataLoaders_DALI/issues/27
    def __next__(self):
            if self._first_batch is not None:
                batch = self._first_batch[0]
                self._first_batch = None
                if self.onehot_label:
                    return [batch[self.output_map[0]], batch[self.output_map[1]].squeeze().long()]
                else:
                    return [batch[self.output_map[0]], batch[self.output_map[1]]]
            data = super().__next__()[0]
            if self.onehot_label:
                return [data[self.output_map[0]], data[self.output_map[1]].squeeze().long()]
            else:
                return [data[self.output_map[0]], data[self.output_map[1]]]
        
    def __len__(self):
        if self.size%self.batch_size==0:
            return self.size//self.batch_size
        else:
            return self.size//self.batch_size+1


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

# pip_train = HybridTrainPipe(batch_size=TRAIN_BS, num_threads=NUM_WORKERS, device_id=0, data_dir=IMG_DIR+'/train', crop=CROP_SIZE, world_size=1, local_rank=0)
# pip_test = HybridValPipe(batch_size=TEST_BS, num_threads=NUM_WORKERS, device_id=0, data_dir=IMG_DIR+'/val', crop=CROP_SIZE, size=VAL_SIZE, world_size=1, local_rank=0)
# train_loader = DALIDataloader(pipeline=pip_train, size=IMAGENET_IMAGES_NUM_TRAIN, batch_size=TRAIN_BS, onehot_label=True)
# test_loader = DALIDataloader(pipeline=pip_test, size=IMAGENET_IMAGES_NUM_TEST, batch_size=TEST_BS, onehot_label=True)