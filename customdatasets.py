import torch
from skimage.io import imread, imsave
from torch.utils import data
import os
import matplotlib.pyplot as plt
from transformations import Compose,Resize,DenseTarget,MoveAxis,Normalize01,AlbuSeg2d
import albumentations
import numpy as np
from libraryNaN import rgb2gray

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.long
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets, repeat(self.pre_transform)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            x, y = imread(input_ID), rgb2gray(imread(target_ID)<255)

        if self.transform is not None:
            x, y = self.transform(x, y)

        # imsave('kern/aug/x/{}.png'.format(index), x[0])
        # imsave('kern/aug/y/{}.png'.format(index), y)

        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp, tar = imread(inp), imread(tar)
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar

# size = (128, 128)
# input_path = 'Kern/img/'
# target_path = 'Kern/mask/'
# inputs = sorted([os.path.join(input_path, name) for name in os.listdir(input_path) if name.endswith('.jpg')])
# targets = sorted([os.path.join(target_path, name) for name in os.listdir(target_path) if name.endswith('.png') and not name.startswith('.')])
# test = SegmentationDataSet(inputs,targets)
