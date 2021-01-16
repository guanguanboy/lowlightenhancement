from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision
import torchvision.transforms as transforms

class SpectralDataSet(Dataset):

    def __init__(self, root_dir, type_name, transform=None):
        self.root_dir = root_dir
        self.type_name = type_name
        self.transform = transform

        clean_path = os.path.join(root_dir, 'clean')
        noise_path = os.path.join(root_dir, 'noise')

        self.path = os.path.join(noise_path, self.type_name)
        self.img_name_list = os.listdir(self.path) #得到path目录下所有图片名称的一个list
        for i in range(20):
            print(self.img_name_list[i])

        self.label_path = os.path.join(clean_path, self.type_name)
        self.label_name_list = os.listdir(self.label_path)
        for i in range(20):
            print(self.label_name_list[i])

    def __getitem__(self, item):
        img_name = self.img_name_list[item]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        #img = img / 512
        label_name = self.label_name_list[item]
        label_item_path = os.path.join(self.label_path, label_name)
        label = Image.open(label_item_path)

        #label = label/512
        if self.transform is not None: #如果transform不等于None,那么执行转换
            img = self.transform(img).float()
            img = img / 512
            label = self.transform(label).float()
            label = label / 512

        return img, label

    def __len__(self):
        return len(self.img_name_list)


def main():

    transform = transforms.Compose([transforms.CenterCrop(256),transforms.ToTensor(),])
    dataset = SpectralDataSet(root_dir='D:/DataSets/hyperspectraldatasets/lowlight_hyperspectral_datasets/band_splited_dataset', type_name='train', transform=transform)
    print(dataset[0][0].shape, dataset[0][1].shape)
    print(len(dataset))

    dataset1 = SpectralDataSet(
        root_dir='D:/DataSets/hyperspectraldatasets/lowlight_hyperspectral_datasets/band_splited_dataset',
        type_name='test', transform=transforms.ToTensor())
    print(dataset1[0])
    print(len(dataset1))

if __name__ == "__main__":
    #main()
    main()
    pass