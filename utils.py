from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    # image_shifted = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def caculate_psnr(image_signal, image_noise):
    pnsr = cal_psnr(image_signal, image_noise) # iamge_signal和iamge_psnr类型要求为ndarray
    return pnsr

def caculate_ssim(image_signal, image_noise):
    ssim = cal_ssim(image_signal, image_noise) # iamge_signal和iamge_psnr类型要求为ndarray
    return ssim

def caculate_psnr_16bit(image_signal, image_noise):
    pnsr = cal_psnr(image_signal, image_noise) # iamge_signal和iamge_psnr类型要求为ndarray
    return pnsr

def caculate_ssim_16bit(image_signal, image_noise):
    ssim = cal_ssim(image_signal, image_noise) # iamge_signal和iamge_psnr类型要求为ndarray
    return ssim