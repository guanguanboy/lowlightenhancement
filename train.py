import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.spectral_dataset import SpectralDataSet
from models.UNet import *
from utils import *
import os
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
input_dim = 1
label_dim = 1
display_step = 20
batch_size = 4
lr = 0.0002
initial_shape = 512
target_shape = 256
device = 'cuda'

def train():
    transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor(), ])
    dataset = SpectralDataSet(root_dir='D:/DataSets/hyperspectraldatasets/lowlight_hyperspectral_datasets/band_splited_dataset', type_name='train', transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    unet = UNet(input_dim, label_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    cur_step = 0

    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the image
            real = real.to(device)
            labels = labels.to(device)

            ### Update U-Net ###
            unet_opt.zero_grad()
            pred = unet(real)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                show_tensor_images(
                    real,
                    size=(input_dim, target_shape, target_shape)
                )
                print('labesl.shape:', labels.shape)
                print('pred.shape:', pred.shape)
                show_tensor_images(labels, size=(label_dim, target_shape, target_shape))
                show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape))
            cur_step += 1


#test_unet = UNet(1, 1)
#assert tuple(test_unet(torch.randn(1, 1, 512, 512)).shape) == (1, 1, 373, 373)

#print("Success!")

train()