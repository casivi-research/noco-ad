import os
import cv2
import math
from utils import colormap
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.metric import *
from utils.coordconv import CoordConv2d
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tools.funcs import rot_img, translation_img, hflip_img, rot90_img, grey_img

from PIL import Image
import torchvision.transforms as T


def standard_normal_loss(embeds):
    # 计算每个特征的均值和方差
    mean = torch.mean(embeds, dim=(1, 2, 3), keepdim=True)
    variance = torch.var(embeds, dim=(1, 2, 3), keepdim=True, unbiased=False)

    # 计算与标准正态分布的均值和方差之间的差异
    target_mean = torch.zeros_like(mean)
    target_variance = torch.ones_like(variance)

    # 使用均方差损失函数
    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(mean, target_mean) + loss_fn(variance, target_variance)

    return loss


def kl_divergence_loss(input_distribution, target_distribution):
    # 添加平滑处理，避免概率值为零
    epsilon = 1e-8
    input_distribution = torch.clamp(input_distribution, min=epsilon)
    target_distribution = torch.clamp(target_distribution, min=epsilon)

    return target_distribution * torch.log(target_distribution / input_distribution)


class CAD_Coarse_Grained(nn.Module):

    def __init__(self, args, nmb_protypes, gamma_d, device):
        super(CAD_Coarse_Grained, self).__init__()

        # init
        self.device = device
        self.args = args

        self.nu = 1
        # self.nu = 1e-3
        self.scale = None

        self.nmb_protypes = nmb_protypes
        self.gamma_d = gamma_d
        self.alpha = 1e-2
        self.K = 1
        self.J = 0
        self.p=2

        self.feature_space = None
        self.centroids = None
        self.history_centroids = None

        self.normal_class = 0

        self.r = nn.Parameter(1e-5 * torch.ones(1), requires_grad=True)
        self.history_r = None

        self.projector = Projector(args.in_dim, args.out_dim).to(self.device)

        self.celoss = nn.CrossEntropyLoss()

        self.best_img_roc = -1
        self.best_pix_roc = -1
        self.best_pix_pro = -1

    def forward(self, embeds):

        features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.centroids.t(), 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(embeds, (self.centroids.t()))
        dist = features + centers - f_c
        dist = torch.sqrt(dist)

        n_neighbors = self.K + self.J
        dist = dist.topk(n_neighbors, largest=False).values

        dissim = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dissim = dissim.unsqueeze(-1)
        score = rearrange(dissim, 'b (h w) c -> b c h w', h=int(math.sqrt(self.args.fp_nums)))

        loss = 0
        if self.training:
            loss = self._soft_boundary(dist)

        return loss, score

    def coarse_forward(self, embeds):
        # embeds=embeds.unsqueeze(dim=-1).unsqueeze(dim=-1)

        embeds = rearrange(embeds, 'b c h w -> b (h w) c')
        
        features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.centroids.t(), 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(embeds, (self.centroids.t()))
        dist = features + centers - f_c
        # dist = torch.cdist(embeds.contiguous(), self.centroids, self.p)
        
        n_neighbors = self.K + self.J
        dist = dist.topk(n_neighbors, largest=False).values
        
        # score = torch.relu(dist - self.r**self.p).squeeze(-1).squeeze(-1)
        score = dist[:, :, 0].unsqueeze(-1)

        loss = 0
        if self.training:
            loss = self._soft_boundary(dist)

        return loss, score

    def _soft_boundary(self, score):
        score = score[:, :self.K] - self.r**self.p
        loss = (1 / self.nu) * torch.relu(score).mean()
        return loss

    def init_banks(self, p, norm, norm_type, train_loader, encoder):
        if norm_type == 'pretrained':
            noise = self.mean_pretrained_c(train_loader, encoder)
            noise = F.normalize(noise, p=p) * norm
        else:
            if norm_type == 'rand':
                noise = torch.rand(self.args.nmb_prototypes[0], self.args.out_dim)
            elif norm_type == 'randn':
                noise = torch.randn(self.args.nmb_prototypes[0], self.args.out_dim)
            elif norm_type == 'ones':
                noise = torch.ones(self.args.nmb_prototypes[0], self.args.out_dim)
            noise = F.normalize(noise, p=p) * norm

        return noise.to(self.device)

    def mean_pretrained_c(self, train_loader, encoder):
        centroids = 0
        with torch.no_grad():
            for it, data in enumerate(train_loader):
                x, labels, sub_category, idx = data
                embeds = encoder.forward_coarse_grained(x.to(self.device))
                embeds = self.projector(embeds)  # multi-scale features
                centroids += embeds.mean(dim=0).mean(dim=-1).mean(dim=-1)

        return centroids[None] / len(train_loader)

    def multiscale_features(self, p):
        embeds = None
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1)
            embeds = o if embeds is None else torch.cat(
                (embeds, F.interpolate(o, embeds.size(2), mode='bilinear')), dim=1)
        return embeds

    def get_assigns(self, embeds, noise_prompts, method='cos'):
        if method == 'l2':
            features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
            centers = torch.sum(torch.pow(noise_prompts.t(), 2), 0, keepdim=True)
            f_c = 2 * torch.matmul(embeds, (noise_prompts.t()))
            dist = features + centers - f_c

            dist = torch.sqrt(dist)
            _, assigns = dist.topk(self.K + self.J, largest=False)  # assigns: torch.Size([8, 3136, 1])

        elif method == 'cos':
            sim = embeds.matmul(noise_prompts.t())
            _, assigns = sim.topk(self.K + self.J, largest=True)

        return assigns.squeeze()


class Descriptor(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Descriptor, self).__init__()
        self.layer = CoordConv2d(in_dim, out_dim, 1)

    def forward(self, embeds):
        # embeds = None
        # for o in embeds:
        #     o = F.avg_pool2d(o, 3, 1, 1)
        #     embeds = o if embeds is None else torch.cat(
        #         (embeds, F.interpolate(o, embeds.size(2), mode='bilinear')), dim=1)

        embeds = self.layer(embeds)

        return embeds

class Projector(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Projector, self).__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.proj = nn.Sequential(
        #     nn.Conv2d(in_dim, 2048, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm2d(2048),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(2048, out_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
        # )
        # self.proj = nn.Sequential(
        #     nn.Linear(in_dim, 2048),
        #     # nn.BatchNorm1d(2048),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(2048, out_dim),
        # )

    def forward(self, embeds):
        # embeds = rearrange(embeds, 'b c h w -> b (h w) c')
        # batch_size, seq_length, emb_dim = embeds.size()
        # embeds = embeds.reshape(batch_size * seq_length, emb_dim)
        embeds = self.proj(embeds)

        # embeds = embeds.reshape(batch_size, -1, 1, 1)

        return embeds

def init_memory(args, init_img_path, encoder, descriptor):
    init_features = 0
    with torch.no_grad():

        pil_img = Image.open(init_img_path).convert('RGB')

        # 使用transforms将PIL图像转换为torch.tensor
        transform_x = T.Compose([
            T.Resize(256, Image.ANTIALIAS),
            T.CenterCrop(args.size_crops[0]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        inp = transform_x(pil_img)[None].cuda(non_blocking=True)
        p = encoder(inp)
        embeds = descriptor(p)
        init_features = rearrange(embeds, 'b c h w -> b (h w) c')

    return init_features[0, :]


def init_cluster(args, init_features):

    i_K, K = 0, args.nmb_prototypes[0]
    with torch.no_grad():

        # 将特征图重塑为56x56x1792的形状
        reshaped_features = init_features.view(-1, args.feat_dim, int(math.sqrt(args.fp_nums)),
                                               int(math.sqrt(args.fp_nums)))
        K_nums = math.ceil(math.sqrt(K))
        centroids = F.avg_pool2d(reshaped_features,
                                 kernel_size=int(math.sqrt(args.fp_nums)) // K_nums,
                                 stride=int(math.sqrt(args.fp_nums)) // K_nums)
        centroids = centroids.view(K, args.feat_dim)

    return centroids
