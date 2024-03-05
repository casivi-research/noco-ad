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
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def trans_tensor_to_np(input_image, std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406]):
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

def CosLoss(data1, data2, Mean=True):
    # stop-gradient
    # data2 = data2.detach()
    cos = nn.CosineSimilarity(dim=1)
    if Mean:
        return 1 - cos(data1, data2).mean()
    else:
        return 1 - cos(data1, data2)


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


def trans_image_cv2pil(img):
    img = Image.fromarray(img)
    img.show()
