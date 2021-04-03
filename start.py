import pathlib
import matplotlib
import numpy as np
import torch
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import util
import os
from inference import predict
from transformations import normalize_01, re_normalize
from unet import UNet
import matplotlib.pyplot as plt
import sys
# np.set_printoptions(threshold=sys.maxsize)
# torch.set_printoptions(threshold=sys.maxsize)
from libraryNaN import images_dir

def maxinimg(img):
    for i in range(0,len(img[0][0])):
        for j in range(0,len(img[0][0][i])):
            one = img[0][0][i][j]
            two = img[0][1][i][j]
            three = img[0][2][i][j]
            four = img[0][3][i][j]
            if two<one and one>three:
                print(one)

size = (1024, 1024)

input_path = 'Photo/test/img/'
target_path = 'Photo/test/mask/'
inputs = images_dir(input_path)
targets = images_dir(target_path)

# read images and store them in memory
images = [imread(img_name) for img_name in inputs]
targets = [imread(tar_name) for tar_name in targets]

sizes = []
for i in images:
    sizes.append((i.shape[0],i.shape[1]))

# Resize images and targets
images_res = [resize(img, (size+(3,))) for img in images]


# device
device = torch.device('cuda')

# model
model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=6,
             start_filters=8,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)


model_name = 'Models/{}/{}.pt'.format('data0326', '56photo_ic3_oc2_nb6_sf8_bs2_3')
model_weights = torch.load(pathlib.Path.cwd() / model_name)
model.load_state_dict(model_weights)

# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)
    img = normalize_01(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img


# postprocess function
def postprocess(img: torch.tensor):

    # img = img[0][2]
    img = torch.argmax(img, dim=1)
    img = img.cpu().numpy()
    img = np.squeeze(img)
    img = re_normalize(img)
    # img = util.invert(img)
    return img


output = [predict(img, model, preprocess, postprocess, device) for img in images_res]

for i, file in enumerate(output):
    file = resize(file, sizes[i], order=0, clip=True, anti_aliasing=False)
    imsave(target_path+os.path.splitext(os.listdir(input_path)[i])[0]+'.png',file)
