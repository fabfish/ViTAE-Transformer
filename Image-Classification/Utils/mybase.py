import logging
import os

# DALI imports

from nvidia.dali.plugin.pytorch import DALIGenericIterator

import os

import nvidia.dali.ops as ops

from torchvision import datasets
from sklearn.utils import shuffle
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms

_logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.49139968, 0.48215827, 0.44653124]
IMAGENET_STD = [0.24703233, 0.24348505, 0.26158768]
IMAGENET_IMAGES_NUM_TRAIN = 1281167
IMAGENET_IMAGES_NUM_TEST = 50000
TRAIN_BS = 256
TEST_BS = 200
NUM_WORKERS = 4
VAL_SIZE = 256
CROP_SIZE = 224

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
        if self.size % self.batch_size==0:
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
