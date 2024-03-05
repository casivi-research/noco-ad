import os
import cv2
import math
from utils import colormap
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from utils.metric import *
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CAD(nn.Module):

    def __init__(self, args, nmb_protypes, gamma_d, device):
        super(CAD, self).__init__()

        # init
        self.device = device
        self.args = args

        self.nu = 1e-3
        self.scale = None

        self.nmb_protypes = nmb_protypes
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 1
        self.J = 0
        self.p=2

        self.feature_space = None
        self.centroids = None
        self.history_centroids = None

        self.r = nn.Parameter(1e-5 * torch.ones(1), requires_grad=True)
        self.history_r = None

        self.projector = Projector(args.in_dim, args.out_dim).to(self.device)

        self.best_img_roc = -1
        self.best_pix_roc = -1
        self.best_pix_pro = -1


    def forward(self, embeds):

        features = torch.sum(torch.pow(embeds, self.p), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.centroids.t(), self.p), 0, keepdim=True)
        f_c = 2 * torch.matmul(embeds, (self.centroids.t()))
        dist = features + centers - f_c
        # dist = torch.cdist(embeds.contiguous(), self.centroids, self.p)
        dist = torch.sqrt(dist)

        n_neighbors = self.K + self.J
        dist = dist.topk(n_neighbors, largest=False).values

        dissim = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dissim = dissim.unsqueeze(-1)
        score = rearrange(dissim, 'b (h w) c -> b c h w', h=int(math.sqrt(self.args.fp_nums)))

        loss = 0
        if self.training:
            loss = (1 / self.nu) * torch.relu(dist[:, :, :self.K] - self.r**self.p).mean()
            # loss=self._soft_boundary2(embeds)
        return loss, score


    def _soft_boundary2(self, embeds):
        features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.centroids.t(), 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(embeds, (self.centroids.t()))
        dist = features + centers - f_c

        n_neighbors = self.K + self.J
        dist = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, :, :self.K] - self.r**2)
        L_att = (1 / self.nu) * torch.relu(score).mean()

        score = (self.r**2 - dist[:, :, self.J:].mean(dim=-1) + dist[:, :, :self.K].mean(dim=-1))
        L_rep = (1 / self.nu) * torch.relu(score - self.alpha).mean()

        loss = L_att + L_rep

        return loss
    
    def init_banks(self, p, norm, norm_type, train_loader, encoder):
        if norm_type == 'pretrained':
            noise = self._mean_pretrained_c(train_loader, encoder)
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

    def multiscale_features(self, p):
        embeds = None
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1)
            embeds = o if embeds is None else torch.cat(
                (embeds, F.interpolate(o, embeds.size(2), mode='bilinear')), dim=1)
        return embeds

    def _mean_pretrained_c(self, train_loader, encoder):
        centroids = 0
        with torch.no_grad():
            for it, (idx, x, _, _, _) in enumerate(train_loader):
                embeds = encoder.forward(x.to(self.device))
                embeds = self.multiscale_features(embeds)
                embeds = self.projector(embeds)  # multi-scale features
                centroids += embeds.mean(dim=0).mean(dim=-1).mean(dim=-1)

        return centroids[None] / len(train_loader)

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


class Projector(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Projector, self).__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.proj = nn.Sequential(
        #     nn.Conv2d(in_dim, 2048, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm2d(2048),
        #     nn.Tanh(inplace=True),
        #     nn.Conv2d(2048, out_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
        # )
        # self.proj = nn.Sequential(
        #     nn.Linear(in_dim, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(2048, out_dim),
        # )

    def forward(self, embeds):
        # embeds = rearrange(embeds, 'b c h w -> b (h w) c')
        # batch_size, seq_length, emb_dim = embeds.size()
        # embeds = embeds.reshape(batch_size * seq_length, emb_dim)
        embeds = self.proj(embeds)

        # embeds = embeds.reshape(batch_size, -1, 56, 56)

        return embeds
