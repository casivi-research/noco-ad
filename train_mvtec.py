import argparse
import shutil
import os
from utils.utils import find_gpus

os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(1)

from utils.metric import *
from utils.visualizer import *

import torch
from cnn.resnet import wide_resnet50_2 as wrn50_2

from datasets.mvtec import MVTEC_CLASS_NAMES, MVTEC_CLASS_NAMES_1, MVTEC_CLASS_NAMES_2, MVTEC_CLASS_NAMES_3
from datasets.mpdd import MPDD_CLASS_NAMES, MPDD_CLASS_NAMES_1, MPDD_CLASS_NAMES_2, MPDD_CLASS_NAMES_3
from datasets.visa import VisA_CLASS_NAMES, VisA_CLASS_NAMES_1, VisA_CLASS_NAMES_2, VisA_CLASS_NAMES_3, VisA_CLASS_NAMES_4, VisA_CLASS_NAMES_5, VisA_CLASS_NAMES_6
from datasets.choose_dataset import choose_datasets

from utils.cad import *
import torch.optim as optim
import warnings
from src.src_utils import (bool_flag, initialize_exp, fix_random_seeds,
                           AverageMeter, create_logger)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Font family not found.*")

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

parser = argparse.ArgumentParser()

#########################
#### data parameters ####
#########################
parser.add_argument("--data_type",
                    type=str,
                    default="mvtec",
                    choices=['mvtec', 'mpdd', 'visa'])
parser.add_argument("--data_path",
                    type=str,
                    default="datasets/MVTec_AD",
                    help="path to dataset repository")
parser.add_argument('--train_class',
                    type=str,
                    default='screw',
                    help='list of train class')
parser.add_argument('--is_train',
                    type=bool,
                    default=True,
                    help='flag on whether using normal only')
parser.add_argument('--fp_nums',
                    type=int,
                    default=56 * 56,
                    help='feature points per image')
parser.add_argument('--Rd', type=bool, default=False)
parser.add_argument("--nmb_crops",
                    type=int,
                    default=[1],
                    nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops",
                    type=int,
                    default=[224],
                    nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument(
    "--min_scale_crops",
    type=float,
    default=[0.14],
    nargs="+",
    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops",
                    type=float,
                    default=[1],
                    nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--crops_for_assign",
                    type=int,
                    nargs="+",
                    default=[0],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature",
                    default=1,
                    type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--feat_dim",
                    default=1792,
                    type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes",
                    default=[1],
                    type=int,
                    nargs="+",
                    help="number of prototypes - it can be multihead")
parser.add_argument("--update_centroids",
                    default='feature_based',
                    type=str,
                    help="method for update centroids",
                    choices=['feature_based', 'learning_based', 'combine'])
parser.add_argument("--do_update",
                    default=False,
                    type=bool,
                    help="flag for update")

parser.add_argument("--m_f",
                    default=0,
                    type=float,
                    help="momentum coefficient for update features")
parser.add_argument("--m_c",
                    default=0,
                    type=float,
                    help="momentum coefficient for update centroids")

#########################
#### optim parameters ###
#########################
parser.add_argument("--use_multi_gpus", default=False, type=bool)

parser.add_argument("--epochs",
                    default=50,
                    type=int,
                    help="number of total epochs to run")

parser.add_argument(
    "--batch_size",
    default=8,
    type=int,
    help="batch size per gpu, i.e. how many unique instances per gpu")

parser.add_argument("--base_lr",
                    default=1e-4,
                    type=float,
                    help="base learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url",
                    default="env://",
                    type=str,
                    help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html"""
                    )
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
                    it is set automatically and should not be passed as argument"""
                    )
parser.add_argument("--local_rank",
                    default=0,
                    type=int,
                    help="this argument is not used and should be ignored")

parser.add_argument("--checkpoint_freq",
                    type=int,
                    default=1,
                    help="Save the model periodically")
parser.add_argument("--seed", type=int, default=1024, help="seed")
args = parser.parse_args()


def main(p=2, margin=1, distribution_type='randn'):

    init_method = 'noise_decoupling'  # choices=[ 'random_image','wo_decoupling', 'noise_decoupling']
    args.update_centroids = 'wo_update'  # choices=[ 'wo_update','feature_based', 'learning_based']

    if args.data_type == 'mvtec':
        args.data_path = "datasets/MVTec_AD"
        args.train_class = MVTEC_CLASS_NAMES
        args.batch_size = 16

    elif args.data_type == 'mpdd':
        args.data_path = "datasets/MPDD"
        args.batch_size = 8
        args.train_class = MPDD_CLASS_NAMES

    elif args.data_type == 'visa':
        args.data_path = "datasets/VisA"
        args.batch_size = 8
        args.train_class = VisA_CLASS_NAMES

    args.cnn_name = 'w_res50'

    args.in_dim = 1792
    args.out_dim = 1792

    num_workers = 8

    args.Rd = False
    args.m_f = 0.8
    args.m_c = 0

    args.checkpoint_freq = args.epochs // 5

    for subcls_idx, subcls_name in enumerate(args.train_class):

        # init
        args.train_class = [subcls_name]
        args.dump_path = f"exp/{distribution_type}_p{p}_norm{margin}/{init_method}/{args.update_centroids}/{args.data_type}"

        logger, _ = initialize_exp(args, "epoch", "loss")
        shutil.copy(os.path.realpath(__file__),
                    os.path.join(args.dump_path, "train.py"))

        # ============= build data ... =============
        train_dataset, test_dataset = choose_datasets(args, subcls_name)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=False,
        )
        logger.info("Train data with {} images loaded.".format(
            len(train_dataset)))
        logger.info("Test data with {} images loaded.".format(
            len(test_dataset)))

        # ============= build model ... =============
        encoder = wrn50_2(pretrained=True, progress=True)
        encoder = encoder.to(device)
        encoder.eval()

        cad = CAD(args, args.nmb_prototypes[0], 1, device)

        # ============= init centroids ... =============
        centroids = cad.init_banks(p, margin, distribution_type, train_loader,
                                   encoder)
        cad.centroids = nn.Parameter(centroids)
        cad.centroids.requires_grad = False
        logger.info(
            f'centroid {distribution_type}_norm is {torch.norm(cad.centroids)}'
        )

        cad = cad.to(device)
        logger.info("Building models done.")

        # ============= buidl optimizer ... =============
        params = [
            {
                'params': cad.parameters()
            },
        ]

        optimizer = optim.AdamW(params=params,
                                lr=args.base_lr,
                                weight_decay=args.wd,
                                amsgrad=True)
        warmup_lr_schedule = np.linspace(0, args.base_lr,
                                         len(train_loader) * 10)
        iters = np.arange(len(train_loader) * (args.epochs - 10))
        cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                            math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
        lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        logger.info("Building optimizer done.")

        # ============= start train ... =============
        best_train_loss = 1e9
        best_image_auroc = 0
        best_pixel_auroc = 0

        cad.best_img_roc = -1
        cad.best_pix_roc = -1
        for epoch in range(args.epochs):
            train_loss = AverageMeter()
            with tqdm(total=len(train_loader)) as t:

                # train
                for it, (idx, x, _, _, _) in enumerate(train_loader):

                    cad.train()
                    optimizer.zero_grad()

                    # forward passes
                    embeds = encoder.forward(x.to(device))
                    embeds = cad.multiscale_features(embeds)
                    embeds = cad.projector(embeds)  # multi-scale features
                    embeds = rearrange(embeds, 'b c h w -> b (h w) c')

                    loss, _, = cad.forward(embeds)

                    loss.backward()
                    optimizer.step()

                    # # flush current state
                    train_loss.update(loss.item(), x.size(0))

                    t.set_postfix({
                        'epoch': f'{epoch}/{args.epochs}',
                        'class': f'{subcls_name}',
                        'do_update': f'{args.update_centroids}',
                        'loss': '{:705.3f}'.format(train_loss.avg),
                        'K': f'{args.nmb_prototypes[0]}'
                    })
                    t.update()

                if it > 0 and it % int(len(train_loader) - 1) == 0:
                    cad.eval()
                    img_roc_auc, per_pixel_rocauc = detection(
                        test_loader, encoder, cad)
                    cad.best_img_roc = max(img_roc_auc, cad.best_img_roc)
                    cad.best_pix_roc = max(per_pixel_rocauc, cad.best_pix_roc)

                    logger.info(' ')
                    logger.info(
                        f'{subcls_name} ' +
                        f'{epoch} - image ROCAUC: {img_roc_auc}|{cad.best_img_roc}'
                    )
                    logger.info(
                        f'{subcls_name} ' +
                        f'{epoch} - pixel ROCAUC: {per_pixel_rocauc}|{cad.best_pix_roc}'
                    )

            if (epoch +
                    1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                logger.info(f'save epoch - {epoch} ckpts.')
                save_dict = {
                    "epoch": epoch + 1,
                    'best_image_aucroc': best_image_auroc,
                    "state_dict": cad.state_dict()
                }
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, f"ckpt-{epoch+1}.pth.tar"),
                )

            if best_image_auroc < cad.best_img_roc:
                best_image_auroc = cad.best_img_roc
                logger.info(
                    f'best_image_aucroc {best_image_auroc} has been found.')
                save_dict = {
                    "epoch": epoch + 1,
                    'best_image_aucroc': best_image_auroc,
                    "state_dict": cad.state_dict()
                }
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, "best_image_aucroc.pth.tar"),
                )


def detection(test_loader, encoder, cad):

    # ============= start train ... =============
    gt_mask_list = list()
    gt_list = list()
    heatmaps = None
    img_roc_auc, per_pixel_rocauc = 0, 0

    w, h = int(math.sqrt(args.fp_nums)), int(math.sqrt(args.fp_nums))

    for _, x, y, mask, _ in tqdm(test_loader):
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())

        x = x.to(device)

        embeds = encoder(x)
        embeds = cad.multiscale_features(embeds)
        embeds = cad.projector(embeds)  # multi-scale features
        embeds = rearrange(embeds, 'b c h w -> b (h w) c')

        _, score = cad.forward(embeds)

        heatmap = score.cpu().detach()
        heatmap = torch.mean(heatmap, dim=1)
        heatmaps = torch.cat(
            (heatmaps, heatmap), dim=0) if heatmaps != None else heatmap

    heatmaps = upsample(heatmaps, size=x.size(2), mode='bilinear')
    heatmaps = gaussian_smooth(heatmaps, sigma=4)

    gt_mask = np.asarray(gt_mask_list)
    scores = rescale(heatmaps)

    fpr, tpr, img_roc_auc = cal_img_roc(scores, gt_list)
    fpr, tpr, per_pixel_rocauc = cal_pxl_roc(gt_mask.astype(int), scores)

    return img_roc_auc, per_pixel_rocauc


if __name__ == '__main__':

    main()
