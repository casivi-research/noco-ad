from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR100
from datasets.base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from utils.utils import trans_tensor_to_pil

import torchvision.transforms as transforms

CIFAR100_CLASS_NAMES = [str(i) for i in range(100)]

class CIFAR100_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 100))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-28.94083453598571, 13.802961825439636), (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628), (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583), (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365), (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279), (-6.132882973622672, 8.046098172351265)]

        transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCIFAR100(root=self.root,
                              train=True,
                              download=True,
                              transform=transform,
                              target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCIFAR100(root=self.root,
                                  train=False,
                                  download=True,
                                  transform=transform,
                                  target_transform=target_transform)


class MyCIFAR100(CIFAR100):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        # img.save(f'{target}.png')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img_pil=trans_tensor_to_pil(img)
        return img, target, index  # only line changed
