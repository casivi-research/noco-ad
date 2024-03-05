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

MVTEC_CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]

MVTEC_CLASS_NAMES_1 = ['cable', 'capsule', 'carpet', 'grid', 'bottle']
MVTEC_CLASS_NAMES_2 = ['screw', 'pill', 'leather', 'metal_nut', 'hazelnut']
MVTEC_CLASS_NAMES_3 = ['tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):

    def __init__(self, dataset_path, class_name='bottle', is_train=True, resize=256, cropsize=224, wild_ver=False):

        assert class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.x, self.y, self.mask = self.load_dataset_folder()

        if wild_ver:
            self.transform_x = A.Compose([
                A.Resize(resize, resize),
                A.Rotate(limit=10),
                A.RandomCrop(cropsize, cropsize),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

            self.transform_mask = A.Compose([
                A.Resize(resize, resize, interpolation=cv2.INTER_NEAREST),
                A.Rotate(limit=10),
                A.RandomCrop(cropsize, cropsize),
                ToTensorV2()
            ])

        else:
            self.transform_x = A.Compose([
                A.Resize(resize, resize),
                A.CenterCrop(cropsize, cropsize),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

            self.transform_mask = A.Compose([
                A.Resize(resize, resize, interpolation=cv2.INTER_NEAREST),
                A.CenterCrop(cropsize, cropsize),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = np.array(x)
        x = self.transform_x(image=x)['image']

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = np.array(mask)
            mask = self.transform_mask(image=mask)['image'] // 255

        return idx, x, y, mask, self.x[idx]

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
