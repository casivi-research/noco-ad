import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from utils.utils import trans_tensor_to_pil

CatsVsDogs_CLASS_NAMES = ['cat', 'dog']


class CatsVsDogsDataset(Dataset):

    def __init__(self,
                 dataset_path,
                 class_name='cat',
                 is_train=True,
                 resize=256,
                 cropsize=224):

        assert class_name in CatsVsDogs_CLASS_NAMES, 'class_name: {}, should be in {}'.format(
            class_name, CatsVsDogs_CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.x, self.y = self.load_dataset_folder()

        self.transform_x = A.Compose([
            A.Resize(resize, resize),
            A.CenterCrop(cropsize, cropsize),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = Image.open(x).convert('RGB')
        x = np.array(x)
        x = self.transform_x(image=x)['image']

        # return idx, x, y

        return x, y, idx
        # return idx, x, y, self.x[idx]

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y = [], []

        if self.is_train:
            img_dir = os.path.join(self.dataset_path, 'train', self.class_name)
        else:
            img_dir = os.path.join(self.dataset_path, 'test')

        for f in os.listdir(img_dir):
            x.append(os.path.join(img_dir, f))
            if f.split('.')[0] == self.class_name:
                y.append(0)
            else:
                y.append(1)
        return x, y
