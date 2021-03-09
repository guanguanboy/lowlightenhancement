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
from torch.optim import lr_scheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

criterion = nn.MSELoss()
n_epochs = 50
input_dim = 1
label_dim = 1
display_step = 20
batch_size = 12
lr = 0.002
target_shape = 256
device = 'cuda'
step_size = 10

def train():
    transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor(), ])
    dataset = SpectralDataSet(root_dir='/mnt/liguanlin/DataSets/lowlight_hyperspectral_datasets/band_splited_dataset', type_name='train', transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)
    unet = UNet(input_dim, label_dim).to(device)
    unet.init_weight()
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(unet_opt, step_size, gamma=0.4)
    cur_step = 0

    for epoch in range(n_epochs):
        train_l_sum, batch_count = 0.0, 0

        for real, labels in tqdm(dataloader):
            #print('real.shape', real.shape)
            #print('labels.shape', labels.shape)
            cur_batch_size = len(real)
            # Flatten the image
            real = real.to(device)
            labels = labels.to(device)

            ### Update U-Net ###
            unet_opt.zero_grad()
            pred = unet(real)
            #print('pred.shape', pred.shape)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

            train_l_sum += unet_loss.cpu().item()
            batch_count += 1

            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                """
                show_tensor_images(
                    real,
                    size=(input_dim, target_shape, target_shape)
                )
                print('labesl.shape:', labels.shape)
                print('pred.shape:', pred.shape)
                show_tensor_images(labels, size=(label_dim, target_shape, target_shape))
                show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape))
                """
            cur_step += 1

        if (epoch + 1) % 2 == 0:
            torch.save(unet.state_dict(), './checkpoints/checkpoint_{}.pth'.format(epoch + 1))

        unet_opt.step() #更新学习率

        print('epoch %d, train loss %.4f' % (epoch + 1, train_l_sum / batch_count))


#test_unet = UNet(1, 1)
#assert tuple(test_unet(torch.randn(1, 1, 512, 512)).shape) == (1, 1, 373, 373)

#print("Success!")

train()