import argparse
import shutil
import os
from utils.utils import find_gpus

os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(1)

from utils.metric import *
from utils.visualizer import *
from utils.colormap import voc_colormap

import torch
from cnn.resnet import wide_resnet50_2 as wrn50_2

from datasets.mvtec import MVTEC_CLASS_NAMES, MVTEC_CLASS_NAMES_1, MVTEC_CLASS_NAMES_2, MVTEC_CLASS_NAMES_3
from datasets.mvtec_loco import MVTEC_LOCO_CLASS_NAMES
from datasets.mpdd import MPDD_CLASS_NAMES, MPDD_CLASS_NAMES_1, MPDD_CLASS_NAMES_2, MPDD_CLASS_NAMES_3
from datasets.visa import VisA_CLASS_NAMES, VisA_CLASS_NAMES_1, VisA_CLASS_NAMES_2, VisA_CLASS_NAMES_3, VisA_CLASS_NAMES_4, VisA_CLASS_NAMES_5, VisA_CLASS_NAMES_6
from datasets.choose_dataset import choose_datasets

from utils.cad import *
import torch.optim as optim
import warnings
from src.src_utils import (bool_flag, initialize_exp, fix_random_seeds, AverageMeter, create_logger)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Font family not found.*")

plt.rcParams['font.family'] = 'DejaVu Sans'

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
parser.add_argument('--fp_nums', type=int, default=56 * 56, help='feature points per image')
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
                    default=[1],
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


class Prototype(nn.Module):

    def __init__(self, input_dim=1792, output_dim=100) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.nmb_prototypes = output_dim
        self.prototypes = nn.Linear(self.input_dim, self.nmb_prototypes, bias=False)

    def forward(self, embeds, K):
        centroids = self.prototypes.weight.data
        dist = torch.cdist(embeds, centroids)  # shape of distances is [B, N, K]
        _, topk_indices = torch.topk(dist, K, dim=-1, largest=False)

        return topk_indices  # [B=4, N=3136, K=3]


def main(p=2, margin=1, distribution_type='randn'):

    # some args args
    global args
    fix_random_seeds(args.seed)

    args.data_type = 'visa'
    init_method = 'noise_decoupling'  # choices=[ 'random_image','wo_decoupling', 'noise_decoupling']
    args.update_centroids = 'wo_update'  # choices=[ 'wo_update','feature_based', 'learning_based']

    if args.data_type == 'mvtec':
        args.data_path = "datasets/MVTec_AD"
        args.train_class = MVTEC_CLASS_NAMES
        args.train_class = ['cable']
        args.batch_size = 1  #16

    elif args.data_type == 'mvtec_loco':
        args.data_path = "datasets/MVTec_LOCO"
        args.train_class = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']

    elif args.data_type == 'mpdd':
        args.data_path = "datasets/MPDD"
        args.batch_size = 1
        args.train_class = ['tubes']
        # args.train_class = MPDD_CLASS_NAMES

    elif args.data_type == 'visa':
        args.data_path = "datasets/VisA"
        args.batch_size = 8
        # args.train_class = ['capsules']
        # args.train_class = VisA_CLASS_NAMES_2
        args.train_class = ['pcb4']
        # args.train_class = VisA_CLASS_NAMES_5 + VisA_CLASS_NAMES_6

    args.nmb_prototypes = [1]

    args.out_dim = 1792

    args.cnn_name = 'w_res50'
    if args.cnn_name == 'w_res50':
        args.in_dim = 1792

    num_workers = min(8, args.batch_size) if args.batch_size > 1 else 1

    args.base_lr = 1e-4
    args.wd = 5e-5
    args.Rd = False
    args.m_f = 0.8
    args.m_c = 0

    # ============= init centroids ... =============
    if args.update_centroids == 'learning_based':
        args.epochs = 50
        args.checkpoint_freq = int(0.1 * args.epochs)
    else:
        args.epochs = 50
        args.checkpoint_freq = args.epochs // 5

    for subcls_idx, subcls_name in enumerate(args.train_class):

        # init
        args.train_class = [subcls_name]
        args.dump_path = f"exp/tmp/sgd/{distribution_type}_p{p}_norm{margin}/{init_method}/{args.update_centroids}/{args.data_type}"

        logger, _ = initialize_exp(args, "epoch", "loss")
        shutil.copy(os.path.realpath(__file__), os.path.join(args.dump_path, "train.py"))

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
        logger.info("Train data with {} images loaded.".format(len(train_dataset)))
        logger.info("Test data with {} images loaded.".format(len(test_dataset)))

        # ============= build model ... =============
        if args.cnn_name == 'w_res50':
            encoder = wrn50_2(pretrained=True, progress=True)

        encoder = encoder.to(device)
        encoder.eval()

        cad = CAD(args, args.nmb_prototypes[0], 1, device)

        # ============= init centroids ... =============
        centroids = cad.init_banks(p, margin, distribution_type, train_loader, encoder)
        cad.centroids = nn.Parameter(centroids)
        logger.info(f'centroid {distribution_type}_norm is {torch.norm(cad.centroids)}')

        cad = cad.to(device)

        if args.update_centroids == 'learning_based':
            cad.centroids.requires_grad = True
        else:
            cad.centroids.requires_grad = False

        logger.info("Building models done.")

        # ============= buidl optimizer ... =============
        params = [
            {
                'params': cad.parameters()
            },
        ]

        # optimizer = optim.SGD(params=params, lr=args.base_lr, momentum=0.9, weight_decay=args.wd)
        optimizer = optim.AdamW(params=params, lr=args.base_lr, weight_decay=args.wd, amsgrad=True)
        warmup_lr_schedule = np.linspace(0, args.base_lr, len(train_loader) * 10)
        iters = np.arange(len(train_loader) * (args.epochs - 10))
        cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                            math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
        lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        logger.info("Building optimizer done.")

        # ============= start train ... =============
        best_train_loss = 1e9
        best_image_auroc = 0
        best_pixel_auroc = 0
        best_pixel_aupro = 0

        cad.best_img_roc = -1
        cad.best_pix_roc = -1
        cad.best_pix_pro = -1
        for epoch in range(args.epochs):
            train_loss = AverageMeter()
            with tqdm(total=len(train_loader)) as t:

                # train
                for it, (idx, x, _, _, _) in enumerate(train_loader):

                    # update learning rate
                    iteration = epoch * len(train_loader) + it
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_schedule[iteration]

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
                    img_roc_auc, per_pixel_rocauc, per_pixel_proauc = detection(test_loader, encoder, cad)
                    cad.best_img_roc = max(img_roc_auc, cad.best_img_roc)
                    cad.best_pix_roc = max(per_pixel_rocauc, cad.best_pix_roc)
                    cad.best_pix_pro = max(per_pixel_proauc, cad.best_pix_pro)

                    logger.info(' ')
                    logger.info(f'{subcls_name} ' + f'{epoch} - image ROCAUC: {img_roc_auc}|{cad.best_img_roc}')
                    logger.info(f'{subcls_name} ' + f'{epoch} - pixel ROCAUC: {per_pixel_rocauc}|{cad.best_pix_roc}')
                    logger.info(f'{subcls_name} ' + f'{epoch} - pixel P_AUPRO: {per_pixel_proauc}|{cad.best_pix_pro}')

            if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                logger.info(f'save epoch - {epoch} ckpts.')
                save_dict = {"epoch": epoch + 1, 'best_image_aucroc': best_image_auroc, "state_dict": cad.state_dict()}
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, f"ckpt-{epoch+1}.pth.tar"),
                )

            if best_image_auroc < cad.best_img_roc:
                best_image_auroc = cad.best_img_roc
                logger.info(f'best_image_aucroc {best_image_auroc} has been found.')
                save_dict = {"epoch": epoch + 1, 'best_image_aucroc': best_image_auroc, "state_dict": cad.state_dict()}
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, "best_image_aucroc.pth.tar"),
                )


def detection(test_loader, encoder, cad):

    # ============= start train ... =============
    gt_mask_list = list()
    gt_list = list()
    heatmaps = None
    img_roc_auc, per_pixel_rocauc, per_pixel_proauc = 0, 0, 0

    w, h = int(math.sqrt(args.fp_nums)), int(math.sqrt(args.fp_nums))

    for _, x, y, mask, _ in tqdm(test_loader):
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())

        x = x.to(device)

        embeds = encoder(x)
        embeds = cad.multiscale_features(embeds)
        embeds = cad.projector(embeds)  # multi-scale features
        embeds = rearrange(embeds, 'b c h w -> b (h w) c')

        _, score= cad.forward(embeds)

        heatmap = score.cpu().detach()
        heatmap = torch.mean(heatmap, dim=1)
        heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap

    heatmaps = upsample(heatmaps, size=x.size(2), mode='bilinear')
    heatmaps = gaussian_smooth(heatmaps, sigma=4)

    gt_mask = np.asarray(gt_mask_list)
    scores = rescale(heatmaps)

    fpr, tpr, img_roc_auc = cal_img_roc(scores, gt_list)
    fpr, tpr, per_pixel_rocauc = cal_pxl_roc(gt_mask.astype(int), scores)
    per_pixel_proauc = cal_pxl_pro(gt_mask, scores)

    return img_roc_auc, per_pixel_rocauc, per_pixel_proauc


def analysis_detection(test_loader, encoder, cad, norm, ckpt_name):

    # ============= start train ... =============
    gt_mask_list = list()
    gt_list = list()
    heatmaps = None
    img_roc_auc, per_pixel_rocauc, per_pixel_proauc = 0, 0, 0
    normal_dist, abnormal_dist = [], []
    normal_norm, abnormal_norm = [], []
    normal_cos, abnormal_cos = [], []

    w, h = int(math.sqrt(args.fp_nums)), int(math.sqrt(args.fp_nums))
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for _, x, y, mask, _ in tqdm(test_loader):
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        mask = torch.nn.functional.interpolate(mask, size=(w, h), mode='nearest')

        x = x.to(device)

        p = encoder(x)
        p = cad.multiscale_features(p)

        embeds = cad.descriptor(p)  # multi-scale features
        p = rearrange(p, 'b c h w -> b (h w) c')
        embeds = rearrange(embeds, 'b c h w -> b (h w) c')

        _, dist, _ = cad.forward(embeds)
        score = rearrange(dist, 'b (h w) c -> b c h w', h=h)

        normal_idx = mask == 0
        normal_idx = rearrange(normal_idx, 'b c h w -> b (h w) c')
        if normal_idx.int().sum() != 0:
            normal_norm += list(torch.norm(embeds[normal_idx[:, :, 0]], dim=-1).cpu().data.numpy())
            normal_dist += list(dist[normal_idx].cpu().data.numpy())
            normal_cos += list(cos(embeds[normal_idx[:, :, 0]], cad.centroids).cpu().data.numpy())

        abnormal_idx = mask == 1
        abnormal_idx = rearrange(abnormal_idx, 'b c h w -> b (h w) c')
        if abnormal_idx.int().sum() != 0:
            abnormal_norm += list(torch.norm(embeds[abnormal_idx[:, :, 0]], dim=-1).cpu().data.numpy())
            abnormal_dist += list(dist[abnormal_idx].cpu().data.numpy())
            abnormal_cos += list(cos(embeds[abnormal_idx[:, :, 0]], cad.centroids).cpu().data.numpy())

        heatmap = score.cpu().detach()
        heatmap = torch.mean(heatmap, dim=1)
        heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap

    heatmaps = upsample(heatmaps, size=x.size(2), mode='bilinear')
    heatmaps = gaussian_smooth(heatmaps, sigma=4)

    gt_mask = np.asarray(gt_mask_list)
    scores = rescale(heatmaps)
    scores = heatmaps

    fpr, tpr, img_roc_auc = cal_img_roc(scores, gt_list)
    fpr, tpr, per_pixel_rocauc = cal_pxl_roc(gt_mask.astype(int), scores)
    per_pixel_proauc = cal_pxl_pro(gt_mask, scores)

    # 设置标记颜色
    normal_color = (86 / 255, 126 / 255, 153 / 255)  # rgb=(86, 126, 153)
    abnormal_color = (179 / 255, 139 / 255, 93 / 255)  # rgb=(179, 139, 93)

    # # 绘制norm的直方图
    plot_histogram(normal_norm, normal_color,
                   os.path.join(cad.args.dump_path, f'Normal_norm Histogram ({norm}_{ckpt_name})'))
    plot_histogram(abnormal_norm, abnormal_color,
                   os.path.join(cad.args.dump_path, f'Abnormal_norm Histogram ({norm}_{ckpt_name})'))
    plot_combine_histogram(normal_norm, normal_color, abnormal_norm, abnormal_color,
                           os.path.join(cad.args.dump_path, f'Combined_Histogram_norm ({norm}_{ckpt_name}).png'))

    # # 绘制cos的直方图
    plot_histogram(normal_cos, normal_color,
                   os.path.join(cad.args.dump_path, f'Normal_cos Histogram ({norm}_{ckpt_name})'))
    plot_histogram(abnormal_cos, abnormal_color,
                   os.path.join(cad.args.dump_path, f'Abnormal_cos Histogram ({norm}_{ckpt_name})'))
    plot_combine_histogram(normal_cos, normal_color, abnormal_cos, abnormal_color,
                           os.path.join(cad.args.dump_path, f'Combined_Histogram_cos ({norm}_{ckpt_name}).png'))

    return img_roc_auc, per_pixel_rocauc, per_pixel_proauc, np.mean(normal_dist), np.mean(abnormal_dist), np.mean(
        normal_norm), np.mean(abnormal_norm), np.mean(normal_cos), np.mean(abnormal_cos)


def plot_combine_histogram(normal_cos, normal_color, abnormal_cos, abnormal_color, title):
    plt.rcParams.update({'font.size': 12})  # 设置全局字体大小为12
    bins = np.linspace(min(min(normal_cos), min(abnormal_cos)), max(max(normal_cos), max(abnormal_cos)),
                       100)  # 使用统一的bins
    normal_hist, _ = np.histogram(normal_cos, bins=bins, density=True)
    abnormal_hist, _ = np.histogram(abnormal_cos, bins=bins, density=True)

    # 将重叠部分以灰色替代
    overlap_hist = np.minimum(normal_hist, abnormal_hist)
    plt.bar(bins[:-1], normal_hist, width=np.diff(bins), color=normal_color, alpha=0.7, label='Normal')
    plt.bar(bins[:-1], abnormal_hist, width=np.diff(bins), color=abnormal_color, alpha=0.7, label='Abnormal')
    plt.bar(bins[:-1], overlap_hist, width=np.diff(bins), color='gray', alpha=0.7, label='Overlap')

    plt.xlabel("Feature Norm")
    plt.ylabel("Density")
    plt.title(title.split('/')[-1])
    plt.legend()
    plt.tight_layout()  # 自动调整布局
    plt.savefig(f'{title}.png')
    plt.clf()


def plot_histogram(data, color, title):
    plt.rcParams.update({'font.size': 12})  # 设置全局字体大小为12
    bins = np.linspace(min(data), max(data), 100)  # 划分100个bin
    hist, _ = np.histogram(data, bins=bins, density=True)
    plt.bar(bins[:-1], hist, width=np.diff(bins), color=color, alpha=0.7)
    plt.xlabel("Feature Norm")
    plt.ylabel("Density")
    plt.title(title.split('/')[-1])
    plt.tight_layout()  # 自动调整布局
    plt.savefig(f'{title}.png')
    plt.clf()
    print(f"已保存{title.split('/')[-1]}")


if __name__ == '__main__':

    main()
