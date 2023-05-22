import torch
import torch.nn.functional as F
import numpy as np
from ssl_utils.train_utils import ce_loss

import torchvision.transforms as transforms
from ssl_utils.augmentation.randaugment import RandAugment

from copy import deepcopy


class MixMatchTransform:
    uratio = 1

    @staticmethod
    def get_transform_weak(transform_base, dataset):
        if dataset == 'seq-cifar10-sscl':
            transform_weak = deepcopy(transform_base)
            transform_weak.transforms.insert(0, transforms.RandomHorizontalFlip())
            transform_weak.transforms.insert(1, transforms.RandomCrop(32, padding=4, padding_mode='reflect'))

            return transform_weak
        else:
            raise NotImplementedError()

    @staticmethod
    def get_transform_strong(transform_base, dataset):
        if dataset == 'seq-cifar10-sscl':
            return MixMatchTransform.get_transform_weak(transform_base, dataset)  # MixMatch只用弱增强
        else:
            raise NotImplementedError()


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def one_hot(targets, nClass, gpu):
    logits = torch.zeros(targets.size(0), nClass).cuda(gpu)
    return logits.scatter_(1, targets.unsqueeze(1), 1)


def mixup_one_target(x, y, gpu, alpha=1.0, is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).cuda(gpu)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam


def consistency_loss(logits_w, y):
    return F.mse_loss(torch.softmax(logits_w, dim=-1), y, reduction='mean')
