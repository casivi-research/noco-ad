import os, torch, cv2, math
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import torch.nn as nn
from torch.nn import init
import random


def freeze_model(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def activate_model(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True


class AverageMeter(object):

    def __init__(self):
        self._val = 0
        self._step = 0

    def update(self, val):
        self._val += val
        self._step += 1

    def __call__(self):
        return self._val / float(self._step)


# class AverageMeter(object):
#     """computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


def find_gpus(nums=4):
    os.system(
        'rm ~/.tmp_free_gpus; touch .tmp_free_gpus; nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >~/.tmp_free_gpus'
    )
    with open(os.path.expanduser('~/.tmp_free_gpus'), 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [(idx, int(x.split()[2]))
                               for idx, x in enumerate(frees)]

    idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
    usingGPUs = [
        str(idx_memory_pair[0])
        for idx_memory_pair in idx_freeMemory_pair[:nums]
    ]
    usingGPUs = ','.join(usingGPUs)
    print('using GPU index: #', usingGPUs)
    return usingGPUs


def fix_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def trans_tensor_to_np(input_image,
                       std=[0.229, 0.224, 0.225],
                       mean=[0.485, 0.456, 0.406]):
    out = np.uint8(
        (input_image.permute(1, 2, 0).detach().cpu().numpy() * std + mean) *
        255)

    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


def trans_tensor_to_pil(input_image,
                        std=[0.229, 0.224, 0.225],
                        mean=[0.485, 0.456, 0.406]):
    out = np.uint8(
        (input_image.permute(1, 2, 0).detach().cpu().numpy() * std + mean) *
        255)
    out = Image.fromarray(out, mode='RGB')
    return out


def save_results(path, loss, epoch, model, optimizer) -> None:
    state_dict = {
        "loss": loss,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state_dict, path)


def warp_image(moving_imgs,
               moving_angle,
               moving_tnx,
               moving_tny,
               device=torch.device('cuda')):
    cos1 = torch.cos(moving_angle * math.pi / 180).unsqueeze(-1)  # 逆时针是正方向
    sin1 = torch.sin(moving_angle * math.pi / 180).unsqueeze(-1)
    affine_matrix = torch.cat([
        torch.cat([cos1, sin1, moving_tnx.unsqueeze(-1)], dim=1).unsqueeze(1),
        torch.cat([-sin1, cos1, moving_tny.unsqueeze(-1)], dim=1).unsqueeze(1)
    ],
                              dim=1).to(device)
    grid = F.affine_grid(affine_matrix,
                         moving_imgs.size(),
                         align_corners=False).float()
    warped_imgs = F.grid_sample(moving_imgs, grid, align_corners=False)
    ones = torch.ones(moving_imgs.size()).to(device)
    warped_masks = F.grid_sample(ones, grid, align_corners=False)
    return warped_imgs, warped_masks


def trans_img_to_train_size(input_image,
                            transform,
                            batch_size,
                            resize_scale,
                            crop_scale,
                            device=torch.device('cuda'),
                            tnx=0,
                            tny=0):
    img_resize = TF.resize(input_image, resize_scale, Image.ANTIALIAS)
    ratio_x = input_image.size[0] / img_resize.size[0]
    ratio_y = input_image.size[1] / img_resize.size[1]
    img = img_resize.crop(box=(img_resize.width / 2 - crop_scale / 2 + tnx,
                               img_resize.height / 2 - crop_scale / 2 + tny,
                               img_resize.width / 2 + crop_scale / 2 + tnx,
                               img_resize.height / 2 + crop_scale / 2 + tny))
    if batch_size != 0:
        img_tensor = transform(img)
        output_image = torch.unsqueeze(img_tensor,
                                       dim=0).expand(batch_size, -1, -1, -1)
        output_image = output_image.to(device)
    else:
        # img = remove_spot(img)
        output_image = transform(img).to(device)
    return output_image, img, ratio_x, ratio_y


def get_rotation_matrix_2D(theta, center, tnx=0, tny=0):
    cx, cy = center
    theta = math.radians(-1 * theta)
    M = np.float32([[
        math.cos(theta), -math.sin(theta),
        (1 - math.cos(theta)) * cx + math.sin(theta) * cy + tnx
    ],
                    [
                        math.sin(theta),
                        math.cos(theta), -math.sin(theta) * cx +
                        (1 - math.cos(theta)) * cy + tny
                    ]])
    return M


def scale_rotate_translate(image, angle=0, displacment=None):
    sr_center = int(-image.size[0] / 2), int(-image.size[1] / 2)
    if displacment is None:
        displacment = 0, 0

    angle = -angle / 180.0 * np.pi

    C = np.array([[1, 0, -sr_center[0]], [0, 1, -sr_center[1]], [0, 0, 1]])

    C_1 = np.linalg.inv(C)

    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    D = np.array([[1, 0, displacment[0]], [0, 1, displacment[1]], [0, 0, 1]])

    Mt = np.dot(np.dot(np.dot(C, R), C_1), D)

    a, b, c = Mt[0]
    d, e, f = Mt[1]

    return image.transform(image.size,
                           Image.AFFINE, (a, b, c, d, e, f),
                           resample=Image.BICUBIC)


def warp_image_PIL(img, angle, tnx, tny):
    margin = np.sqrt(pow(tnx, 2) + pow(tny, 2))
    direction = np.arctan2(tny, -tnx)
    displacment = (margin * np.cos(np.pi / 4 - direction),
                   margin * np.sin(np.pi / 4 - direction))
    dst_img = scale_rotate_translate(img, angle,
                                     (img.size[0] / 2, img.size[1] / 2),
                                     displacment)
    return dst_img


def check_dir(path_list: list):
    for path in path_list:
        if os.path.exists(path) == False:
            os.mkdir(path)

    return None


def save_image(image, save_path):
    if isinstance(image, np.ndarray):
        # opencv格式图像
        cv2.imwrite(save_path, np.uint8(image))
    elif isinstance(image, Image.Image):
        # PIL格式图像
        image.save(save_path)
    else:
        raise TypeError('pic should be PIL Image or ndarray.')

    return None


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight,
                                         ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        loss = torch.cuda.FloatTensor([0])
        for idx in range(preds.shape[0]):
            logpt = -self.ce_fn(preds[idx].unsqueeze(0),
                                labels[idx].unsqueeze(0))
            pt = torch.exp(logpt)
            loss += -((1 - pt)**self.gamma) * self.alpha * logpt

        return loss.mean()


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) +
                    smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # loss = 1 - loss.sum() / N
        return 1 - loss.mean()


def CosLoss(data1, data2, Mean=True):
    # stop-gradient
    # data2 = data2.detach()
    cos = nn.CosineSimilarity(dim=1)
    if Mean:
        return 1 - cos(data1, data2).mean()
    else:
        return 1 - cos(data1, data2)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


class CriterionPixelWise(nn.Module):

    def __init__(self, ignore_index=255):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.sm = nn.Softmax2d()

    def forward(self, preds_S, preds_T):
        n, c, w, h = preds_T.shape
        logit_s = torch.log(self.sm(preds_S)).view(n, -1)
        p_t = self.sm(preds_T).view(n, -1)
        loss = F.kl_div(logit_s, p_t)
        return loss


def get_IOU(gt_mask, pre_mask):
    intersection = cv2.bitwise_and(gt_mask, pre_mask)
    union = cv2.bitwise_or(gt_mask, pre_mask)
    diff = (union - gt_mask)

    intersection_lx, intersection_ly = (intersection[:, :, 0] != 0).nonzero()
    union_lx, union_ly = (union[:, :, 0] != 0).nonzero()
    diff_lx, diff_ly = (diff[:, :, 0] != 0).nonzero()

    jac_img = gt_mask
    jac_img[intersection_lx, intersection_ly] = [30, 34, 156]
    jac_img[diff_lx, diff_ly] = [62, 64, 234]

    jac = 0
    if len(union_lx) != 0:
        jac = len(intersection_lx) / len(union_lx)

    return jac_img, jac


def cal_mIoU(seg, gt, classes=3, background_id=0):
    channel_iou = []
    for i in range(classes):
        if i == background_id:
            continue
        cond = i**2
        # 计算相交部分
        if len(np.where(gt == cond)[0]) == 0:
            continue

        inter = len(np.where(seg * gt == cond)[0])

        union = len(np.where(seg == i)[0]) + len(np.where(gt == i)[0]) - inter
        if union == 0:
            iou = 0
        else:
            iou = inter / union
        channel_iou.append(iou)
    res = np.array(channel_iou).mean()
    return res


def Get_Heatmap(fmps, src_size):
    heatmap = torch.sum(fmps, dim=0).unsqueeze(dim=0)
    max_value = torch.max(heatmap)
    min_value = torch.min(heatmap)
    heatmap = (heatmap - min_value) / (max_value - min_value) * 255
    heatmap = heatmap.detach().cpu().numpy().astype(np.uint8).transpose(
        1, 2, 0)  # 尺寸大小，如：(45, 45, 1)
    heatmap = cv2.resize(heatmap, (src_size, src_size),
                         interpolation=cv2.INTER_LINEAR)  # 重整图片到原尺寸
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def np_2_pil(img):
    img = Image.fromarray(img)
    # img.show()
    return img
