from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import torch
import pandas as pd
from pathlib import Path


images_path = Path('data/jpg/')
TRAIN_LABELS_PATH = Path('data/trainlabels.csv')
VAL_LABELS_PATH = Path('data/vallabels.csv')
TEST_LABELS_PATH = Path('data/testlabels.csv')


class ImagesDataset(Dataset):
    def __init__(self, labels_file, transform=None):
        self.img_labels = pd.read_csv(labels_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_id, label = self.img_labels.iloc[index]
        img_filename = f'image_{img_id:05}.jpg'
        img = read_image(images_path / img_filename).to(torch.float32) / 255.0
        if self.transform:
            img = self.transform(img)
        
        label -= 1

        return img, label


vgg19_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

yolov5_transform = transforms.Compose([
    transforms.Resize((640, 640)),
])


if __name__ == '__main__':
    images_path = Path('../data/jpg/')
    # Testing re-sizing works
    print('Without Resize')
    f = ImagesDataset('../data/imagelabels.csv')
    print(f[0][0].shape)
    print(f[1][0].shape)
    print(f[2][0].shape)

    print('VGG19 Resize')
    f = ImagesDataset('../data/imagelabels.csv', vgg19_transform)
    print(f[0][0].shape)
    print(f[1][0].shape)
    print(f[2][0].shape)

    print('YOLOv5 Resize')
    f = ImagesDataset('../data/imagelabels.csv', yolov5_transform)
    print(f[0][0].shape)
    print(f[1][0].shape)
    print(f[2][0].shape)
