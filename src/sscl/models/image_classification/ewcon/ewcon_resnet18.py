#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
EWC Online
"""

import torch
from torch import nn
import torch.nn.functional as F
from mmcls.models import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier

from sscl.models import BaseLearningMethod
from tqdm import tqdm


@CLASSIFIERS.register_module(force=True)
class EwconResnet18(ImageClassifier, BaseLearningMethod):
    ALG_NAME = "EwconResnet18"  # 算法模型的名称
    ALG_COMPATIBILITY = ["class-il", "task-il", "domain-il"]

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(EwconResnet18, self).__init__(
                 backbone,
                 neck=neck,
                 head=head,
                 pretrained=pretrained,
                 train_cfg=train_cfg,
                 init_cfg=init_cfg
        )

        self.e_lambda = 10
        self.gamma = 1.0

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None

        if not hasattr(self, "device"):
            self.device = torch.device("cpu")

    def get_logits(self, img):
        x = self.extract_feat(img)
        x = self.head.pre_logits(x)
        cls_score = self.head.fc(x)
        return cls_score

    # 一些对模型参数进行调整的算法（例如基于正则化的oEWC）可能会需要下面的接口：
    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        return torch.cat(self.get_grads_list())

    def sup_forward_train(self, img, gt_label, **kwargs):
        """
        该方法来自 mmcls\models\classifiers\image.py 中 ImageClassifier，
        因为需要额外返回 logits，所以重写该方法
        :param img:
        :param gt_label:
        :param kwargs:
        :return:
        """

        losses = {}

        """
        为了提取 logits，需要分解 loss = self.head.forward_train(x, gt_label)，
        参考 mmcls\models\heads\linear_head.py 中的 LinearClsHead
        """
        cls_score = self.get_logits(img)
        loss = self.head.loss(cls_score, gt_label)
        losses.update(loss)

        return losses, cls_score

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish * ((self.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def after_task(self, dataloaders, runner, *args, **kwargs):
        train_loader = dataloaders[0]
        fish = torch.zeros_like(self.get_params())

        cnt = 0
        with tqdm(total=len(train_loader)) as pbar:
            for j, data in enumerate(train_loader):
                cnt += len(data['gt_label'])
                inputs, labels = data['img'], data['gt_label']
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                for ex, lab in zip(inputs, labels):
                    runner.optimizer.zero_grad()
                    output = self.get_logits(ex.unsqueeze(0))
                    loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                        reduction='none')
                    exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                    loss = torch.mean(loss)
                    loss.backward()
                    fish += exp_cond_prob * self.get_grads() ** 2
                pbar.update(1)

        fish /= cnt

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.gamma
            self.fish += fish

        self.checkpoint = self.get_params().data.clone()

    def ewcon_forward_train(self):
        penalty = self.penalty()
        loss = self.e_lambda * penalty
        assert not torch.isnan(loss)
        return loss

    def forward_train(self, img, gt_label, **kwargs):
        """

        :param img: (N, C, H, W)
        :param gt_label: (N, 1)
        :param kwargs: 包含了 img_metas 和 unsup 样本数据，但当前方法没有用到
                kwargs['unsup']['weak'][0] = {
                    'img_metas': [img_meta_dict, img_meta_dict, ...],
                    'img': N*C*H*W 的 tensor
        :return:
        """

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        self.device = img.device

        losses = {}
        losses_sup, logits = self.sup_forward_train(img, gt_label, **kwargs)
        losses.update(losses_sup)

        loss_ewcon = self.ewcon_forward_train()
        losses.update({
            "ewcon_loss": loss_ewcon
        })

        return losses
