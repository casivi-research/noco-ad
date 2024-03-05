import random
import argparse
import shutil
import os
from utils.utils import find_gpus, freeze_model, CosLoss, trans_tensor_to_np

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
from datasets.cifar10 import CIFAR10_CLASS_NAMES, CIFAR10_CLASS_NAMES_0, CIFAR10_CLASS_NAMES_1, CIFAR10_CLASS_NAMES_2, CIFAR10_CLASS_NAMES_3, CIFAR10_CLASS_NAMES_4, cifar_map_dict
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


def main(p=2, norm=1.0, norm_type='randn'):

    # some args args
    global args
    fix_random_seeds(args.seed)

    args.data_type = 'fashion-mnist'
    init_method = 'noise_decoupling'  # choices=[ 'random_image','wo_decoupling', 'noise_decoupling']
    args.update_centroids = 'wo_update'  # choices=[ 'wo_update','feature_based', 'learning_based']

    if args.data_type == 'fashion-mnist':
        args.data_path = "datasets"
        # args.train_class = ['0', '1','2','3','4','5','6','7','8','9']
        args.train_class = ['8']

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

    args.epochs = 20
    args.checkpoint_freq = int(0.1 * args.epochs)

    for subcls_idx, subcls_name in enumerate(args.train_class):

        # init
        args.train_class = [subcls_name]
        normal_class = int(args.train_class[0])
        args.dump_path = f"exp/randn_p{p}_norm{norm}/resnet152/{args.data_type}"
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
        centroids = cad_coarse.init_banks(p, norm, norm_type, train_loader, encoder)
        cad_coarse.centroids = centroids
        cad_coarse.centroids = nn.Parameter(centroids)
        cad_coarse.centroids = nn.Parameter(cad_coarse.centroids, requires_grad=False)

        cad_coarse.normal_class = normal_class

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
        best_train_loss = 1e9
        best_image_auroc = 0
        best_pixel_auroc = 0
        best_pixel_aupro = 0

        cad_coarse.best_img_roc = -1
        cad_coarse.best_pix_roc = -1
        cad_coarse.best_pix_pro = -1
        for epoch in range(args.epochs):
            train_loss = AverageMeter()
            with tqdm(total=len(train_loader)) as t:

                # train
                for it, data in enumerate(train_loader):
                    x, _, _ = data
                    x = x.to(device)  # shape:[8, 3, 32, 32]

                    cad_coarse.train()
                    optimizer.zero_grad()

                    embeds = encoder.forward_coarse_grained(x)
                    embeds = cad_coarse.projector(embeds)

                    loss, _ = cad_coarse.coarse_forward(embeds)

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
                    if it > 0 and it % int(len(train_loader) - 1) == 0:
                        cad_coarse.eval()
                        img_roc_auc, normal_dist, abnormal_dist = coarse_detection(test_loader, encoder, cad_coarse)
                        cad_coarse.best_img_roc = max(img_roc_auc, cad_coarse.best_img_roc)
                        logger.info(
                            f'{args.train_class[0]} ' + f'epoch: {epoch} - it: {it} - ' +
                            f'image AUROC: {img_roc_auc}|{cad_coarse.best_img_roc} - ' +
                            f'ratio: {abnormal_dist/normal_dist} - normal_dist avg: {normal_dist} - abnormal_dist avg: {abnormal_dist} - '
                            + f'loss: {train_loss.avg}')

            #         if best_image_auroc < cad_coarse.best_img_roc:
            #             best_image_auroc = cad_coarse.best_img_roc
            #             logger.info(f'best_image_aucroc {best_image_auroc} has been found.')
            #             save_dict = {
            #                 "epoch": epoch + 1,
            #                 "iter": it + 1,
            #                 'best_image_aucroc': best_image_auroc,
            #                 "state_dict": cad_coarse.state_dict()
            #             }
            #             torch.save(
            #                 save_dict,
            #                 os.path.join(args.dump_path, "best_image_aucroc.pth.tar"),
            #             )

            # if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            #     logger.info(f'save epoch - {epoch} ckpts.')
            #     save_dict = {
            #         "epoch": epoch + 1,
            #         'best_image_aucroc': best_image_auroc,
            #         "state_dict": cad_coarse.state_dict()
            #     }
            #     torch.save(
            #         save_dict,
            #         os.path.join(args.dump_path, f"ckpt-{epoch+1}.pth.tar"),
            #     )


def coarse_detection(test_loader, encoder, cad_coarse):
    img_roc_auc = 0
    idx_label_score = []
    normal_dist, abnormal_dist = [], []
    with torch.no_grad():
        for data in tqdm(test_loader):
            x, labels, idx = data
            x = x.to(device)  # shape:[8, 3, 32, 32]

            # forward passes
            embeds = encoder.forward_coarse_grained(x)
            embeds = cad_coarse.projector(embeds)

            _, score = cad_coarse.coarse_forward(embeds)

            normal_dist += list(score[labels == 0].cpu().data.numpy())
            abnormal_dist += list(score[labels == 1].cpu().data.numpy())

            idx_label_score += list(
                zip(idx.cpu().data.numpy().tolist(),
                    labels.cpu().data.numpy().tolist(),
                    score.cpu().data.numpy().tolist()))

        _, labels, scores = zip(*idx_label_score)

        labels = np.array(labels)
        scores = np.array(scores)

        img_roc_auc = roc_auc_score(labels, scores)

    return img_roc_auc, np.mean(normal_dist), np.mean(abnormal_dist)


if __name__ == '__main__':
    main()