import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from utils.utils import trans_tensor_to_pil

VisA_CLASS_NAMES_1 = ['candle', 'capsules']
VisA_CLASS_NAMES_2 = ['cashew', 'chewinggum']
VisA_CLASS_NAMES_3 = ['fryum', 'macaroni1']
VisA_CLASS_NAMES_4 = ['macaroni2', 'pcb1']
VisA_CLASS_NAMES_5 = ['pcb2', 'pcb3']
VisA_CLASS_NAMES_6 = ['pcb4', 'pipe_fryum']

VisA_CLASS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
    'pipe_fryum'
]

class VisADataset(Dataset):

    def __init__(self, dataset_path, class_name='candle', resize=256, cropsize=224, is_train=True, wild_ver=False):
        assert class_name in VisA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, VisA_CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y, self.mask = self.load_dataset_folder()

        if wild_ver:
            self.transform_x = T.Compose([
                T.RandomRotation(10),
                T.Resize(resize, Image.LANCZOS),
                T.RandomCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.transform_mask = T.Compose(
                [T.RandomRotation(10),
                 T.Resize(resize, Image.NEAREST),
                 T.RandomRotation(10),
                 T.RandomCrop(cropsize),
                 T.ToTensor()])

        else:
            self.transform_x = T.Compose([
                T.Resize(resize, Image.LANCZOS),
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST), T.CenterCrop(cropsize), T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = np.uint8(np.array(mask) > 0) * 255
            mask = Image.fromarray(mask)
            mask = self.transform_mask(mask)

        return idx, x, y, mask, self.x[idx]

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self, sample_nums=300):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        normal_dir = os.path.join(self.dataset_path, self.class_name, 'Data/Images/Normal')
        abnormal_dir = os.path.join(self.dataset_path, self.class_name, 'Data/Images/Anomaly')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'Data/Masks/Anomaly')

        normal_list = sorted(os.listdir(normal_dir))
        abnormal_list = sorted(os.listdir(abnormal_dir))

        # split normal imgs into train_normal_list and test_normal_list
        train_normal_list = normal_list[:-100]
        # if sample_nums < len(train_normal_list):
        #     train_normal_list = train_normal_list[:sample_nums]

        test_normal_list = normal_list[-100:]

        if self.is_train:
            for f in train_normal_list:
                if f.endswith('.JPG'):
                    x.append(os.path.join(normal_dir, f))
                    y.append(0)
                    mask.append(None)
        else:
            for f in test_normal_list:
                if f.endswith('.JPG'):
                    x.append(os.path.join(normal_dir, f))
                    y.append(0)
                    mask.append(None)
            for f in abnormal_list:
                if f.endswith('.JPG'):
                    x.append(os.path.join(abnormal_dir, f))
                    y.append(1)
                    mask.append(os.path.join(gt_dir, f.split('.')[0] + '.png'))

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)
