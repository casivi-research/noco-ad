from datasets.mvtec import MVTecDataset
from datasets.mpdd import MPDD_Dataset
from datasets.visa import VisADataset
from datasets.btad import BTADDataset
from datasets.mvtec_loco import MVTec_LOCO_Dataset
from datasets.cifar10 import CIFAR10_Dataset
from datasets.cifar100 import CIFAR100_Dataset
from datasets.mnist import MNIST_Dataset
from datasets.fashion import FashionMNIST_Dataset
from datasets.cats_and_dogs import CatsVsDogsDataset


def choose_datasets(args, train_class, noise=None):
    if args.data_type == 'mvtec':
        train_dataset = MVTecDataset(dataset_path=args.data_path,
                                     class_name=train_class,
                                     is_train=True,
                                     resize=256,
                                     cropsize=args.size_crops[0],
                                     wild_ver=False)

        test_dataset = MVTecDataset(dataset_path=args.data_path,
                                    class_name=train_class,
                                    is_train=False,
                                    resize=256,
                                    cropsize=args.size_crops[0],
                                    wild_ver=args.Rd)
    elif args.data_type == 'mpdd':
        train_dataset = MPDD_Dataset(dataset_path=args.data_path,
                                     class_name=train_class,
                                     is_train=True,
                                     resize=256,
                                     cropsize=args.size_crops[0],
                                     wild_ver=False)

        test_dataset = MPDD_Dataset(dataset_path=args.data_path,
                                    class_name=train_class,
                                    is_train=False,
                                    resize=256,
                                    cropsize=args.size_crops[0],
                                    wild_ver=args.Rd)
    elif args.data_type == 'visa':
        train_dataset = VisADataset(dataset_path=args.data_path,
                                    class_name=train_class,
                                    is_train=True,
                                    resize=256,
                                    cropsize=args.size_crops[0],
                                    wild_ver=False)

        test_dataset = VisADataset(dataset_path=args.data_path,
                                   class_name=train_class,
                                   is_train=False,
                                   resize=256,
                                   cropsize=args.size_crops[0],
                                   wild_ver=args.Rd)
    elif args.data_type == 'cifar10':
        train_dataset = CIFAR10_Dataset(root=args.data_path, normal_class=int(train_class[0]))
        test_dataset = None

    elif args.data_type == 'cifar100':
        train_dataset = CIFAR100_Dataset(root=args.data_path, normal_class=int(train_class[0]))
        test_dataset = None

    elif args.data_type == 'mnist':
        train_dataset = MNIST_Dataset(root=args.data_path, normal_class=int(train_class[0]))
        test_dataset = None

    elif args.data_type == 'fashion-mnist':
        train_dataset = FashionMNIST_Dataset(root=args.data_path, normal_class=int(train_class[0]))
        test_dataset = None

    elif args.data_type == 'cats_and_dogs':
        train_dataset = CatsVsDogsDataset(dataset_path=args.data_path,
                                          class_name=train_class,
                                          is_train=True,
                                          resize=256,
                                          cropsize=args.size_crops[0])

        test_dataset = CatsVsDogsDataset(dataset_path=args.data_path,
                                         class_name=train_class,
                                         is_train=False,
                                         resize=256,
                                         cropsize=args.size_crops[0])

    else:
        raise NotImplementedError('unsupport dataset type')

    return train_dataset, test_dataset
