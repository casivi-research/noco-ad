# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import random
from logging import getLogger
from PIL import ImageFilter, Image

from PIL import ImageFilter
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import transforms as T

from torch.utils.data import Dataset, DataLoader

logger = getLogger()

CLASS_NAMES_1 = ['capsule', 'carpet', 'grid']
# CLASS_NAMES_2 = ['hazelnut', 'leather', 'metal_nut', 'pill']
CLASS_NAMES_2 = ['pill', 'screw']
# CLASS_NAMES_3 = ['tile','toothbrush','transistor','wood','zipper']
CLASS_NAMES_3 = ['zipper']
CLASS_NAMES = [
    'bottle',
    'cable',
    'capsule',
    'carpet',
    'grid',
    'hazelnut',
    'leather',
    'metal_nut',
    'pill',
    'screw',
    'tile',
    'toothbrush',
    'transistor',
    'wood',
    'zipper',
]


class MultiCropDataset(datasets.ImageFolder):

    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([
                transforms.Compose([
                    randomresizedcrop,
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Compose(color_transform),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class MultiCropMvtec(DataLoader):

    def __init__(
            self,
            data_path,
            size_crops,  # 224
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            size_dataset=-1,
            return_index=False,
            train_class=['screw'],
            is_train=True):
        super(MultiCropMvtec, self).__init__(data_path)

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        self.is_train = is_train
        self.return_index = return_index
        self.samples = self.load_imgs_to_list(data_path, train_class)

        # ------------------- some trans -------------------
        trans = []
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([
                transforms.Compose([
                    # randomresizedcrop,
                    transforms.Resize(size_crops[i]),
                    transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.Compose(color_transform),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
                ])
            ] * nmb_crops[i])
        self.trans = trans

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = Image.open(path).convert('RGB')  # [64,64]
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops

    def load_imgs_to_list(self, data_path, train_class) -> list:
        imgs_path_list = []
        prefix_list = ['jpg', 'png', 'bmp']

        for _class in train_class:
            # add subclass whithin train/ and test/ to phase list
            if self.is_train:
                phase = ['train/good']
            else:
                phase = [
                    os.path.join('test', defect_type)
                    for defect_type in os.listdir(os.path.join(data_path, _class, 'test'))
                ]

            # scan every file within phase folder
            for _phase in phase:
                path = os.path.join(data_path, _class, _phase)
                for file in os.listdir(path):
                    if file.split('.')[-1] in prefix_list:
                        imgs_path_list.append((os.path.join(path, file), train_class.index(_class)))

        return imgs_path_list


class MVTecDataset(Dataset):

    def __init__(self, dataset_path, class_name='bottle', is_train=True, resize=256, cropsize=224, wild_ver=False):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y, self.mask = self.load_dataset_folder()

        if wild_ver:
            self.transform_x = T.Compose([
                T.Resize(resize, Image.ANTIALIAS),
                T.RandomRotation(10),
                T.RandomCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.transform_mask = T.Compose(
                [T.Resize(resize, Image.NEAREST),
                 T.RandomRotation(10),
                 T.RandomCrop(cropsize),
                 T.ToTensor()])

        else:
            self.transform_x = T.Compose([
                T.Resize(resize, Image.ANTIALIAS),
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
            mask = self.transform_mask(mask)

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


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
