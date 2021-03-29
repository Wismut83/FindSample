import os
from transformations import Compose, AlbuSeg2d, DenseTarget, MoveAxis, Normalize01, Resize
from customdatasets import SegmentationDataSet
from torch.utils.data import DataLoader
from skimage.io import imread, imsave
from libraryNaN import images_dir, renumerate

size = (512, 512)
input_path = 'Photo/learn/4class/img/'
target_path = 'Photo/learn/4class/mask/'
inputs1 = images_dir(input_path)
targets1 = images_dir(target_path)

for i, inputs in enumerate(targets1): # inputs1
    img, max = renumerate(imread(inputs), max=True)
    print(inputs,'\t',max, '\t', img.shape)

    # targets = [targets1[i]]
    # transforms_training = Compose([
    #     # Resize(input_size=size+(3,), target_size=(size)),
    #     DenseTarget(),
    #     MoveAxis(),
    #     Normalize01()
    # ])
    #
    # dataset_train = SegmentationDataSet(inputs=[inputs],
    #                                     targets=targets,
    #                                     transform=transforms_training)
    #
    # dataloader_training = DataLoader(dataset=dataset_train,
    #                                  batch_size=2,
    #                                  shuffle=True)
    #
    # x, y = next(iter(dataloader_training))
    # print(f'x = shape: {x.shape}; type: {x.dtype}')
    # print(f'x = min: {x.min()}; max: {x.max()}')
    # print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')
    # print(f'y = min: {y.min()}; max: {y.max()}\n')