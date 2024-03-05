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


class Noise_VPT(nn.Module):

    def __init__(self, args, nmb_protypes, gamma_d, device):
        super(Noise_VPT, self).__init__()

        # init
        self.device = device
        self.args = args

        self.nu = 1e-3
        self.scale = None

        self.nmb_protypes = nmb_protypes
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 3
        self.J = 0

        self.feature_space = None
        self.centroids = None
        self.history_centroids = None

        self.r = nn.Parameter(1e-5 * torch.ones(1), requires_grad=False)
        self.history_r = None

        self.descriptor = Descriptor(args.feat_dim, args.feat_dim).to(self.device)

        self.celoss = nn.CrossEntropyLoss()

        self.best_img_roc = -1
        self.best_pix_roc = -1
        self.best_pix_pro = -1

    def ce_forward(self, embeds):

        n_neighbors = 6
        self.celoss = nn.CrossEntropyLoss()

        dot_product = embeds.matmul(self.centroids.t())
        sim = torch.nn.Softmax(dim=-1)(dot_product)
        sim, _ = sim.topk(n_neighbors, largest=True)

        labels = torch.zeros(sim.size(0), sim.size(1)).long().to(self.device)
        labels[:, 0] = 1

        loss = self.celoss(sim.view(-1, n_neighbors), labels.view(-1))

        dissim = (1 - sim)[:, :, 0]
        dissim = dissim.unsqueeze(-1)
        score = rearrange(dissim, 'b (h w) c -> b c h w', h=int(math.sqrt(self.args.fp_nums)))
        return loss, score, embeds

    def forward(self, embeds, centroids):
        features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(centroids.t(), 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(embeds, (centroids.t()))
        dist = torch.sqrt(features + centers - f_c)
        dist = dist.topk(self.K + self.J, largest=False).values

        dissim = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dissim = dissim.unsqueeze(-1)
        score = rearrange(dissim, 'b (h w) c -> b c h w', h=int(math.sqrt(self.args.fp_nums)))

        return score

    def _soft_boundary(self, embeds, centroids, assigns):
        features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(centroids.t(), 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(embeds, (centroids.t()))
        dist = torch.sqrt(features + centers - f_c)
        
        # dist = dist.topk(self.K + self.J, largest=False).values 
        dist=torch.gather(dist, 2, assigns)

        score = (dist[:, :, :self.K] - self.r**2)
        loss = (1 / self.nu) * torch.relu(score).mean()

        return loss

    def init_banks(self, init_method, encoder=None, data_loader=None):
        ##################### init centroid banks ... #####################
        if init_method == 'noise_decoupling':
            # noise = torch.rand(self.args.nmb_prototypes[0], self.args.feat_dim)
            noise = torch.randn(self.args.nmb_prototypes[0], self.args.feat_dim)
            # noise = torch.ones(self.args.nmb_prototypes[0], self.args.feat_dim)
            # noise = torch.sparse(self.args.nmb_prototypes[0], self.args.feat_dim)
            # noise = torch.eye(self.args.nmb_prototypes[0], self.args.feat_dim)

            # noise = F.normalize(noise)
            # return noise.to(self.device), None

            Q, R = torch.qr(noise.t())
            return Q.t().to(self.device), None

        elif init_method == 'random_image':
            centoids, mean_f = self._init_centroid(encoder, data_loader)
            return centoids.to(self.device), mean_f.to(self.device)
        else:
            raise NotImplementedError(f'{init_method} is an invalid init-method.')

    def _init_centroid(self, encoder, data_loader):
        for i, (_, x, _, _, _) in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            p = encoder(x)
            self.scale = p[0].size(2)
            phi_p = self.descriptor(p)
            mean_f = torch.mean(phi_p, dim=0, keepdim=True).detach()
            break

        mean_f = rearrange(mean_f, 'b c h w -> (b h w) c').cpu().detach().numpy()
        centroids = KMeans(n_clusters=self.args.nmb_prototypes[0], max_iter=3000).fit(mean_f).cluster_centers_

        centroids = torch.Tensor(centroids)
        mean_f = torch.Tensor(mean_f)

        return centroids, mean_f

    def multiscale_features(self, p):
        embeds = None
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1)
            embeds = o if embeds is None else torch.cat(
                (embeds, F.interpolate(o, embeds.size(2), mode='bilinear')), dim=1)
        return embeds

    def get_assigns(self, embeds, noise_prompts, method='cos'):
        embeds=F.normalize(embeds,dim=-1)
        noise_prompts=F.normalize(noise_prompts, dim=-1)
        if method == 'l2':
            features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
            centers = torch.sum(torch.pow(self.centroids.t(), 2), 0, keepdim=True)
            f_c = 2 * torch.matmul(embeds, (self.centroids.t()))
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
