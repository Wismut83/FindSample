import pathlib
from transformations import Compose, AlbuSeg2d, DenseTarget
from transformations import MoveAxis, Normalize01, Resize
from sklearn.model_selection import train_test_split
from customdatasets import SegmentationDataSet
import torch
from unet import UNet
from trainer import Trainer
from torch.utils.data import DataLoader
import albumentations
import os
from torchsummary import summary
import matplotlib.pyplot as plt
import math
from libraryNaN import images_dir

size = (1024, 1024)
inputs = images_dir('Photo/learn/2class/img/')
targets = images_dir('Photo/learn/2class/mask/')

# training transformations and augmentations
transforms_training = Compose([
    Resize(input_size=size + (3,), target_size=(size)),
    AlbuSeg2d(albu=albumentations.HorizontalFlip(p=0.5)),
    # AlbuSeg2d(albu=albumentations.Rotate(p=0.5, limit=4)),
    AlbuSeg2d(albu=albumentations.VerticalFlip(p=0.5)),
    # AlbuSeg2d(albu=albumentations.RGBShift(p=0.5)),
    AlbuSeg2d(albu=albumentations.RandomGamma(p=0.5)),
    AlbuSeg2d(albu=albumentations.ShiftScaleRotate(p=0.5,
                                                   rotate_limit=3,
                                                   shift_limit=0.03,
                                                   scale_limit=0.03, )),
    DenseTarget(),
    MoveAxis(),
    Normalize01()
])

transforms_validation = Compose([
    Resize(input_size=size + (3,), target_size=(size)),
    DenseTarget(),
    MoveAxis(),
    Normalize01()])

random_seed = 1337

train_size = 0.8

inputs_train, inputs_valid = train_test_split(
    inputs,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

targets_train, targets_valid = train_test_split(
    targets,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

dataset_train = SegmentationDataSet(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training)

dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation)

dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=2,
                                 shuffle=True)

dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=2,
                                   shuffle=True)

device = torch.device('cuda')

model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=6,
             start_filters=16,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)

# summary(model, ((3,)+size))

print(f'train ({len(dataset_train.inputs)}): {dataset_train.inputs}')
print(f'valid ({len(dataset_valid.inputs)}): {dataset_valid.inputs}')

# model_name_one = 'Models/{}/{}.pt'.format('data0326', '45photo_ic3_oc2_nb6_sf16_bs2')
# model_weights = torch.load(pathlib.Path.cwd() / model_name_one)
# model.load_state_dict(model_weights)

epochs = 300

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000003)  # 0.0002

# lambda1 = lambda epoch: epoch // 30
# lambda2 = lambda epoch: 0.95 ** epoch
# lambda3 = lambda epoch: 1 / (1 + 0.05 * epoch)
# lambda4 = lambda epoch: abs(math.asin(math.sin(epoch/7))/(0.5*math.pi))
# lambda5 = lambda epoch: abs(math.asin(math.sin((epoch-math.pi*40/2)/(epochs/10)))/(0.5*math.pi))*(epochs-epoch)/epochs
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda5)
# (cos(pi*x*2)+1)*(10-x)/10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs // 6, eta_min=0)

trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=scheduler,
                  epochs=epochs,
                  epoch=0,
                  notebook=False)

training_losses, validation_losses, lr_rates = trainer.run_trainer()
model_name = 'Models/{}/{}.pt'.format('data0326', '45photo_ic3_oc2_nb6_sf16_bs2')
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

from lr_rate_finder import LearningRateFinder

lrf = LearningRateFinder(model, criterion, optimizer, device)
lrf.fit(dataloader_training, steps=100)
lrf.plot()

from visual import plot_training

fig = plot_training(training_losses, validation_losses, lr_rates, gaussian=False, sigma=1, figsize=(10, 4))
plt.show()
