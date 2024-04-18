import random
import argparse
import shutil
import os
from utils.utils import find_gpus, freeze_model, CosLoss, trans_tensor_to_np
import gc  # 导入垃圾回收模块

os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(1)

from utils.metric import *
from utils.visualizer import *
from utils.colormap import voc_colormap, mvtec_colormap

import torch
import torchvision
from cnn.resnet import wide_resnet50_2, resnet18, resnet152

from datasets.mvtec import MVTEC_CLASS_NAMES, MVTEC_CLASS_NAMES_1, MVTEC_CLASS_NAMES_2, MVTEC_CLASS_NAMES_3
from datasets.mvtec_loco import MVTEC_LOCO_CLASS_NAMES
from datasets.mpdd import MPDD_CLASS_NAMES, MPDD_CLASS_NAMES_1, MPDD_CLASS_NAMES_2, MPDD_CLASS_NAMES_3
from datasets.visa import VisA_CLASS_NAMES, VisA_CLASS_NAMES_1, VisA_CLASS_NAMES_2, VisA_CLASS_NAMES_3, VisA_CLASS_NAMES_4, VisA_CLASS_NAMES_5, VisA_CLASS_NAMES_6
from datasets.cifar10 import CIFAR10_CLASS_NAMES, CIFAR10_CLASS_NAMES_0, CIFAR10_CLASS_NAMES_1, CIFAR10_CLASS_NAMES_2, CIFAR10_CLASS_NAMES_3, CIFAR10_CLASS_NAMES_4
from datasets.cifar100 import CIFAR100_CLASS_NAMES
from datasets.choose_dataset import choose_datasets

from utils.cad_coarse_grained import *
import torch.optim as optim
import warnings
from src.src_utils import (bool_flag, initialize_exp, fix_random_seeds, AverageMeter, create_logger)

import logging

warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

parser = argparse.ArgumentParser()

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="datasets/MVTec_AD", help="path to dataset repository")
parser.add_argument("--data_type", type=str, default="mvtec", choices=['mvtec', 'mpdd', 'visa', 'btad'])
parser.add_argument('--train_class', type=str, default='screw', help='list of train class')
parser.add_argument('--is_train', type=bool, default=True, help='flag on whether using normal only')
parser.add_argument('--fp_nums', type=int, default=8 * 8, help='feature points per image')
parser.add_argument('--Rd', type=bool, default=False)

parser.add_argument("--nmb_crops", type=int, default=[1], nargs="+", help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+", help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops",
                    type=float,
                    default=[0.14],
                    nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops",
                    type=float,
                    default=[1],
                    nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## dcv2 specific args #
#########################
parser.add_argument("--crops_for_assign",
                    type=int,
                    nargs="+",
                    default=[0],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=1, type=float, help="temperature parameter in training loss")
parser.add_argument("--feat_dim", default=1792, type=int, help="feature dimension")
parser.add_argument("--nmb_prototypes",
                    default=[10],
                    type=int,
                    nargs="+",
                    help="number of prototypes - it can be multihead")
parser.add_argument("--update_centroids",
                    default='feature_based',
                    type=str,
                    help="method for update centroids",
                    choices=['feature_based', 'learning_based', 'combine'])
parser.add_argument("--do_update", default=False, type=bool, help="flag for update")

parser.add_argument("--m_f", default=0, type=float, help="momentum coefficient for update features")
parser.add_argument("--m_c", default=0, type=float, help="momentum coefficient for update centroids")

#########################
#### optim parameters ###
#########################
parser.add_argument("--use_multi_gpus", default=False, type=bool)

parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
parser.add_argument("--batch_size",
                    default=1,
                    type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=1e-1, type=float, help="base learning rate")
parser.add_argument("--start_warmup", default=0, type=float, help="initial warmup learning rate")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url",
                    default="env://",
                    type=str,
                    help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size",
                    default=1,
                    type=int,
                    help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank",
                    default=0,
                    type=int,
                    help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int, help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=1, type=int, help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=1, help="Save the model periodically")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size",
                    type=int,
                    default=8,
                    help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path",
                    type=str,
                    default="online_deep_cfa_decoder_cfa_wo_updateC",
                    help="experiment dump path for checkpoints and log")
parser.add_argument(
    "--ckpts_path",
    type=str,
    default=
    'online_deep_cfa/screw/baselr1.0_p10_e500_b1_r224x224_d1792_(2023_03_23_22_35)/checkpoints/best_train_loss.pth',
    help="path of pretrained model")

parser.add_argument("--do_vis", type=bool, default=False, help="")
parser.add_argument("--vis_type", type=str, default='odc', help="odc or kmeans")
parser.add_argument("--assign_vis", type=str, default='assign_vis', help="odc or kmeans")
parser.add_argument("--seed", type=int, default=1024, help="seed")
args = parser.parse_args()


def main(p=2, margin=1.0, distribution_type='randn'):
    # some args args
    global args
    fix_random_seeds(args.seed)

    args.data_type = 'cifar10'
    init_method = 'noise_decoupling'  # choices=[ 'random_image','wo_decoupling', 'noise_decoupling']
    args.update_centroids = 'wo_update'  # choices=[ 'wo_update','feature_based', 'learning_based']

    if args.data_type == 'cifar10':
        args.data_path = "datasets"
        args.train_class = CIFAR10_CLASS_NAMES
        # args.train_class = ['1']

    if args.data_type == 'cifar100':
        args.data_path = "datasets"
        args.train_class = ['0']
        # args.train_class = [str(i) for i in range(100)]

    if args.data_type == 'cats_and_dogs':
        args.data_path = "datasets/CatsVSDogs/0.2-0.8"
        # args.train_class = ['cat', 'dog']
        args.train_class = ['dog']

    args.out_dim = 2048
    args.cnn_name = 'res152'
    if args.cnn_name == 'res152':
        args.in_dim = 2048

    args.batch_size = 64
    args.nmb_prototypes = [1]
    args.num_workers = min(8, args.batch_size) if args.batch_size > 1 else 1

    args.base_lr = 1e-3
    args.wd = 5e-4
    args.Rd = False

    args.m_f = 0.99
    args.m_c = 0

    args.epochs = 10
    args.checkpoint_freq = 1

    for subcls_idx, subcls_name in enumerate(args.train_class):

        # init
        args.train_class = [subcls_name]
        args.dump_path = f"exp/{distribution_type}_p{p}_norm{margin}/resnet152/{args.data_type}"
        # args.dump_path = f"exp/{distribution_type}_p{p}_wto_norm/resnet152/{args.data_type}"
        logger, _ = initialize_exp(args, "epoch", "loss")
        shutil.copy(os.path.realpath(__file__), os.path.join(args.dump_path, "snapshot_train.py"))
        shutil.copy('utils/cad_coarse_grained.py', os.path.join(args.dump_path, "snapshot_model.py"))

        # ============= build data ... =============
        train_dataset, test_dataset = choose_datasets(args, subcls_name)

        if args.data_type in ['cifar10', 'cifar100']:
            train_loader, test_loader = train_dataset.loaders(batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                pin_memory=True,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=False,
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                pin_memory=True,
                num_workers=args.num_workers,
                drop_last=False,
            )

        logger.info("Train data with {} images loaded.".format(args.batch_size * len(train_loader)))
        logger.info("Test data with {} images loaded.".format(args.batch_size * len(test_loader)))

        # ============= build model ... =============
        encoder = resnet152(pretrained=True, progress=True)
        encoder = encoder.to(device)
        encoder.eval()

        cad_coarse = CAD_Coarse_Grained(args, args.nmb_prototypes[0], 1, device)

        # ============= init centroids ... =============
        centroids = cad_coarse.init_banks(p, margin, distribution_type, train_loader, encoder)
        cad_coarse.centroids = nn.Parameter(centroids)
        cad_coarse.normal_class = subcls_name

        if args.update_centroids == 'learning_based':
            cad_coarse.centroids = nn.Parameter(cad_coarse.centroids, requires_grad=True)
        else:
            cad_coarse.centroids = nn.Parameter(cad_coarse.centroids, requires_grad=False)

        cad_coarse.p = p
        logger.info(f'centroid {distribution_type}_{p}-norm is {torch.norm(cad_coarse.centroids,p)}')

        cad_coarse = cad_coarse.to(device)
        logger.info("Building models done.")

        # ============= buidl optimizer ... =============
        params = [
            {
                'params': cad_coarse.parameters()
            },
        ]
        optimizer = optim.AdamW(params=params, lr=1e-3, weight_decay=5e-4, amsgrad=True)
        logger.info("Building optimizer done.")

        # ============= start train ... =============
        cad_coarse.best_img_roc = -1
        best_image_auroc = -1

        for epoch in range(args.epochs):
            train_loss = AverageMeter()
            with tqdm(total=len(train_loader)) as t:

                # train
                for it, data in enumerate(train_loader):
                    # _, x, _, _ = data
                    x, labels, _, idx = data
                    x = x.to(device)  # shape:[8, 3, 32, 32]

                    cad_coarse.train()
                    optimizer.zero_grad()

                    embeds = encoder.forward_coarse_grained(x)
                    embeds = cad_coarse.projector(embeds)
                    loss, _ = cad_coarse.coarse_forward(embeds)

                    loss += cad_coarse.r * 1e3

                    loss.backward()
                    optimizer.step()

                    # # flush current state
                    train_loss.update(loss.item(), x.size(0))

                    t.set_postfix({
                        'epoch': f'{epoch}/{args.epochs}',
                        'class': f'{args.train_class[0]}',
                        'do_update': f'{args.update_centroids}',
                        'loss': '{:705.3f}'.format(train_loss.avg),
                        'K': f'{args.nmb_prototypes[0]}'
                    })
                    t.update()

                    # if epoch % 1 == 0:
                    # if it > 0 and it % 5 == 0:
                    if it > 0 and (it % int(5 + epoch) == 0 or it % int(len(train_loader) - 1) == 0):
                        # if it > 0 and it % int(len(train_loader) - 1) == 0:
                        cad_coarse.eval()
                        img_roc_auc = coarse_detection(test_loader, encoder, cad_coarse)

                        cad_coarse.best_img_roc = max(img_roc_auc, cad_coarse.best_img_roc)
                        logger.info(f'{args.train_class[0]} - epoch: {epoch} - it: {it} - ' +
                                    f'image AUROC: {img_roc_auc}|{cad_coarse.best_img_roc} - loss: {train_loss.avg}')

                        # logger.info(
                        #     f'{args.train_class[0]} - epoch: {epoch} - it: {it} - ' +
                        #     f'dist_ratio: {abnormal_dist/normal_dist} - normal_dist: {normal_dist} - abnormal_dist: {abnormal_dist}'
                        # )

                        # logger.info(
                        #     f'{args.train_class[0]} - epoch: {epoch} - it: {it} - ' +
                        #     f'normal_ratio: {abnormal_norm/normal_norm}, normal_norm: {normal_norm}, abnormal_norm: {abnormal_norm}'
                        # )

                    if best_image_auroc < cad_coarse.best_img_roc:
                        best_image_auroc = cad_coarse.best_img_roc
                        logger.info(f'best_image_aucroc {best_image_auroc} has been found.')
                        save_dict = {
                            "epoch": epoch + 1,
                            "iter": it + 1,
                            'best_image_aucroc': best_image_auroc,
                            "state_dict": cad_coarse.state_dict()
                        }
                        torch.save(
                            save_dict,
                            os.path.join(args.dump_path, "best_image_aucroc.pth.tar"),
                        )

            if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                logger.info(f'save epoch - {epoch} ckpts.')
                save_dict = {
                    "epoch": epoch + 1,
                    'best_image_aucroc': best_image_auroc,
                    "state_dict": cad_coarse.state_dict()
                }
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, f"ckpt-{epoch+1}.pth.tar"),
                )


def norm_analysis():

    # some args args
    global args
    fix_random_seeds(args.seed)

    args.data_type = 'cifar10'
    init_method = 'noise_decoupling'  # choices=[ 'random_image','wo_decoupling', 'noise_decoupling']
    args.update_centroids = 'wo_update'  # choices=[ 'wo_update','feature_based', 'learning_based']

    if args.data_type == 'cifar10':
        args.data_path = "datasets"
        args.train_class = ['0']
        # args.train_class = CIFAR10_CLASS_NAMES

    args.out_dim = 512
    args.cnn_name = 'res152'
    if args.cnn_name == 'res152':
        args.in_dim = 2048

    args.batch_size = 32 * 2
    args.nmb_prototypes = [1]
    args.num_workers = min(8, args.batch_size) if args.batch_size > 1 else 1

    args.Rd = False

    args.epochs = 1
    # args.checkpoint_freq = int(0.1 * args.epochs)

    for subcls_idx, subcls_name in enumerate(args.train_class):

        # init
        args.train_class = [subcls_name]
        normal_class = int(args.train_class[0])
        args.dump_path = f"exp/norm_analysis/tmp/randn_32.0*/{args.cnn_name}/{init_method}/{args.update_centroids}/{args.data_type}"
        logger, _ = initialize_exp(args, "epoch", "loss")
        shutil.copy(os.path.realpath(__file__), os.path.join(args.dump_path, "train.py"))

        # ============= build data ... =============
        dataset, _ = choose_datasets(args, subcls_name)
        train_loader, test_loader = dataset.loaders(batch_size=args.batch_size, num_workers=args.num_workers)
        logger.info("Train data with {} images loaded.".format(args.batch_size * len(train_loader)))
        logger.info("Test data with {} images loaded.".format(args.batch_size * len(test_loader)))

        # ============= build model ... =============
        # encoder = resnet18(pretrained=True, progress=True)
        # encoder = wide_resnet50_2(pretrained=True, progress=True)
        encoder = resnet152(pretrained=True, progress=True)
        encoder = encoder.to(device)
        encoder.eval()

        # ============= init centroids ... =============
        cad_coarse = CAD_Coarse_Grained(args, args.nmb_prototypes[0], 1, device)
        centroids, _ = cad_coarse.init_banks(init_method)
        cad_coarse.centroids = centroids
        # cad_coarse.centroids = nn.Parameter(centroids)
        cad_coarse.normal_class = normal_class

        if args.update_centroids == 'learning_based':
            cad_coarse.centroids = nn.Parameter(cad_coarse.centroids, requires_grad=True)
        else:
            cad_coarse.centroids = nn.Parameter(cad_coarse.centroids, requires_grad=False)

        cad_coarse = cad_coarse.to(device)
        logger.info("Building models done.")

        # ============= load pretrained ckpts ... =============
        # ckpts_dir = f'exp/multic/randn_normalize_1.0*/noise_decoupling/wo_update/mvtec/screw/baselr0.001_p10_e30_b8_r224x224_d1792_mf0.8_mc0_(2023_07_16_19_47_19)'
        # ckpts_path = os.path.join(ckpts_dir, 'best_image_aucroc.pth.tar')

        ckpts_dir = f'exp/norm_analysis/randn_32.0*/resnet152/noise_decoupling/wo_update/cifar10/{args.train_class[0]}'
        for f in os.listdir(ckpts_dir):
            ckpts_path = os.path.join(ckpts_dir, f, 'best_image_aucroc.pth.tar')

        state_dict = torch.load(ckpts_path)
        cad_coarse.load_state_dict(state_dict['state_dict'])
        logger.info(f"Load pretrained_ckpts from {ckpts_path}.")

        # ============= start ... =============
        img_roc_auc, normal_dist, abnormal_dist, normal_norm, abnormal_norm = coarse_detection(
            test_loader, encoder, cad_coarse)

        logger.info(f'{ckpts_dir}')
        logger.info(f'img_roc_auc: {img_roc_auc}')

        logger.info(
            f'dist_ratio: {abnormal_dist/normal_dist}, normal_dist: {abnormal_dist}, normal_dist: {abnormal_dist}')

        logger.info(
            f'normal_ratio: {abnormal_norm/normal_norm}, normal_norm: {normal_norm}, abnormal_norm: {abnormal_norm}')


def coarse_detection(test_loader, encoder, cad_coarse):
    img_roc_auc = 0
    idx_label_score = []
    # normal_dist, abnormal_dist = [], []
    # normal_norm, abnormal_norm = [], []
    with torch.no_grad():
        for data in tqdm(test_loader):
            # idx, x, labels, _ = data
            x, labels, _, idx = data
            x = x.to(device)  # shape:[8, 3, 32, 32]

            # forward passes
            embeds = encoder.forward_coarse_grained(x)
            embeds = cad_coarse.projector(embeds)

            # normal_idx = labels == 0
            # abnormal_idx = labels == 1

            # if normal_idx.int().sum() != 0:
            #     normal_norm += list(torch.norm(embeds[normal_idx]).cpu().data.numpy()[None])
            # if abnormal_idx.int().sum() != 0:
            #     abnormal_norm += list(torch.norm(embeds[abnormal_idx]).cpu().data.numpy()[None])

            _, score = cad_coarse.coarse_forward(embeds)

            # normal_dist += list(score[labels == 0].cpu().data.numpy())
            # abnormal_dist += list(score[labels == 1].cpu().data.numpy())

            idx_label_score += list(
                zip(idx.cpu().data.numpy().tolist(),
                    labels.cpu().data.numpy().tolist(),
                    score.cpu().data.numpy().tolist()))

        _, labels, scores = zip(*idx_label_score)

        labels = np.array(labels)
        scores = np.array(scores)

        img_roc_auc = roc_auc_score(labels, scores)

    return img_roc_auc
    # return img_roc_auc, np.mean(normal_dist), np.mean(abnormal_dist), np.mean(normal_norm), np.mean(abnormal_norm)


if __name__ == '__main__':
    # for margin in [1.0, 32, 16, 0.5, 8, 4]:
    #     main(margin=margin)

    # for distribution_type in ['rand', 'ones', 'pretrained']:
    #     main(distribution_type=distribution_type)

    main()
