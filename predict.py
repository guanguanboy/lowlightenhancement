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
from torchvision.utils import save_image
from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


criterion = nn.MSELoss()
n_epochs = 100
input_dim = 1
label_dim = 1
display_step = 20
batch_size = 12
lr = 0.002
target_shape = 256

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    unet = UNet(input_dim, label_dim)
    unet.load_state_dict(torch.load('./checkpoints_1_19/checkpoint_20.pth', map_location='cpu'))

    unet.eval()

    if use_cuda:
        unet.cuda()

    criterion = nn.MSELoss()

    """
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    # 读取训练数据集
    batch_size = 512

    #准备数据
    mnist_test_dataset_with_noise = MyMnistDataSet.MyMnistDataSet(root_dir='./mnist_dataset_noise',
                                                                  label_root_dir='./mnist_dataset', type_name='test',
                                                                  transform=transforms.ToTensor())
    test_data_loader_with_noise = torch.utils.data.DataLoader(mnist_test_dataset_with_noise, batch_size, shuffle=False,
                                                              num_workers=num_workers)
    #遍历数据已有模型进行reference
    test_loss_sum, batch_count, start_time = 0.0, 0, time.time()
    for X, y in test_data_loader_with_noise:

        X = X.to(device)
        y = y.to(device)

        y_hat = unet(X)

        l = criterion(y_hat, y)

        test_loss_sum += l.cpu().item()
        batch_count += 1

    print('predict: batch_cout %d, test loss %.4f, time %.1f sec'
          % (batch_count, test_loss_sum / batch_count, time.time() - start_time))
    """



    transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor(), ])
    dataset = SpectralDataSet(root_dir='/mnt/liguanlin/DataSets/lowlight_hyperspectral_datasets/band_splited_dataset', type_name='test', transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)

    for real, labels in tqdm(dataloader):
        #print('real.shape', real.shape)
        #print('labels.shape', labels.shape)
        cur_batch_size = len(real)
        # Flatten the image
        real = real.to(device)
        labels = labels.to(device)

        pred = unet(real)

        print(pred.shape)
        print(real.shape)
        save_image(pred, 'pred.png', nrow=2)
        save_image(labels, 'labels.png', nrow=2)

        pred_numpy = pred.detach().cpu().numpy()
        label_numpy = labels.detach().cpu().numpy()
        origin_numpy = real.detach().cpu().numpy()

        print(pred_numpy.shape)

        for i in range(pred_numpy.shape[0]):
            numpy_img = pred_numpy[i].reshape((pred_numpy.shape[2], pred_numpy.shape[3]))
            numpy_img = numpy_img * 1023
            numpy_img = numpy_img.astype(np.int16)
            image = Image.fromarray(numpy_img)
            iamge_name = "./test_results/pred/" + str(i) + ".png"
            image.save(iamge_name)

            label_numpy_img = label_numpy[i].reshape((label_numpy.shape[2], label_numpy.shape[3]))
            label_numpy_img = label_numpy_img * 1023
            label_numpy_img = label_numpy_img.astype(np.int16)
            label_image = Image.fromarray(label_numpy_img)
            label_iamge_name = "./test_results/label/" + str(i) + ".png"
            label_image.save(label_iamge_name)

            origin_numpy_img = origin_numpy[i].reshape((origin_numpy.shape[2], origin_numpy.shape[3]))
            origin_numpy_img = origin_numpy_img * 1023
            origin_numpy_img = origin_numpy_img.astype(np.int16)
            origin_image = Image.fromarray(origin_numpy_img)
            origin_iamge_name = "./test_results/original/" + str(i) + ".png"
            origin_image.save(origin_iamge_name)

        break

if __name__=="__main__":
    main()