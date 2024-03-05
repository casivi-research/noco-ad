from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import FashionMNIST
from datasets.base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
import torchvision.transforms as transforms
from utils.utils import trans_tensor_to_pil


class FashionMNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyFashionMNIST(root=self.root,
                                   train=True,
                                   download=True,
                                   transform=transform,
                                   target_transform=target_transform)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyFashionMNIST(root=self.root,
                                       train=False,
                                       download=True,
                                       transform=transform,
                                       target_transform=target_transform)


class MyFashionMNIST(FashionMNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyFashionMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_rgb = np.repeat(img.numpy()[:, :, np.newaxis], 3, axis=2)
        img = Image.fromarray(img_rgb)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
