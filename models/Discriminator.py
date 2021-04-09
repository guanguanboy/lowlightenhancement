import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)

from models.UNetGenerator import *


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: Discriminator
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        #### START CODE HERE ####
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x, y):
        #print('x.shape:', x.shape)
        #print('y.shape:', y.shape)
        x = torch.cat([x, y], axis=1)
        #print('x.shape after cat:', x.shape)
        x0 = self.upfeature(x)
        #print('x0.shape:',x0.shape)
        x1 = self.contract1(x0)
        #print('x1.shape:',x1.shape)
        x2 = self.contract2(x1)
        #print('x2.shape:',x2.shape)
        x3 = self.contract3(x2)
        #print('x3.shape:',x3.shape)
        x4 = self.contract4(x3)
        #print('x4.shape:',x4.shape)
        xn = self.final(x4)
        #print('xn.shape:',xn.shape)
        return xn

# UNIT TEST
test_discriminator = Discriminator(10, 1)
assert tuple(test_discriminator(
    torch.randn(1, 5, 256, 256), 
    torch.randn(1, 5, 256, 256)
).shape) == (1, 1, 16, 16)
print("Success!")