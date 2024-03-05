import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.metric import *
from utils.coordconv import CoordConv2d
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets.mvtec as mvtec
import datasets.mpdd as mpdd
from datasets.mvtec import MVTecDataset
from datasets.mpdd import MPDD_Dataset


class CAD(nn.Module):

    def __init__(self, params, logger, train_class, train_loader, model, nmb_protypes, gamma_d, device):
        super(CAD, self).__init__()
        
        # init
        self.device = device
        self.logger = logger
        self.params = params
        self.train_class = train_class

        self.centroids = 0
        self.nu = 1e-3
        self.scale = None

        self.nmb_protypes = nmb_protypes
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 3
        self.J = 3

        self.r = nn.Parameter(1e-5 * torch.ones(1), requires_grad=True)
        self.Descriptor = Descriptor(self.gamma_d).to(device)

        self._init_centroid(model, train_loader)

        self.best_img_roc = -1
        self.best_pix_roc = -1
        self.best_pix_pro = -1

    def forward(self, p):

        embeds = self.Descriptor(p)
        embeds = rearrange(embeds, 'b c h w -> b (h w) c')

        features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.centroids, 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(embeds, (self.centroids))
        dist = features + centers - f_c
        dist = torch.sqrt(dist)

        n_neighbors = self.K
        dist = dist.topk(n_neighbors, largest=False).values

        dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dist = dist.unsqueeze(-1)

        score = rearrange(dist, 'b (h w) c -> b c h w', h=self.scale)

        loss = 0
        if self.training:
            loss = self._soft_boundary(embeds)

        return loss, score

    def _soft_boundary(self, phi_p):
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.centroids, 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(phi_p, (self.centroids))
        dist = features + centers - f_c
        n_neighbors = self.K + self.J
        dist = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, :, :self.K] - self.r**2)
        L_att = (1 / self.nu) * torch.relu(score).mean()

        score = (self.r**2 - dist[:, :, self.J:].mean(dim=-1) + dist[:, :, :self.K].mean(dim=-1))
        L_rep = (1 / self.nu) * torch.relu(score - self.alpha).mean()

        # score = (dist[:, : , :self.K] - self.r**2)
        # L_att = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))

        # score = (self.r**2 - dist[:, : , self.J:])
        # L_rep  = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))

        loss = L_att + L_rep

        return loss

    def _init_centroid(self, model, data_loader):
        for i, (x, _, _) in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            p = model(x)
            self.scale = p[0].size(2)
            phi_p = self.Descriptor(p)
            self.centroids = ((self.centroids * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (i + 1)

        self.centroids = rearrange(self.centroids, 'b c h w -> (b h w) c').detach()
        if self.nmb_protypes > 1:
            self.centroids = self.centroids.cpu().detach().numpy()
            self.centroids = KMeans(n_clusters=self.nmb_protypes, max_iter=3000).fit(self.centroids).cluster_centers_
            self.centroids = torch.Tensor(self.centroids).to(self.device)

        self.centroids = self.centroids.transpose(-1, -2).detach()
        self.centroids = nn.Parameter(self.centroids, requires_grad=False)


class Descriptor(nn.Module):

    def __init__(self, gamma_d):
        super(Descriptor, self).__init__()
        dim = 1792
        self.layer = CoordConv2d(dim, dim // gamma_d, 1)\

    def forward(self, p):
        sample = None
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1)
            sample = o if sample is None else torch.cat(
                (sample, F.interpolate(o, sample.size(2), mode='bilinear')), dim=1)

        embeds = self.layer(sample)
        # embeds = rearrange(embeds, 'b c h w -> b (h w) c')
        
        return embeds