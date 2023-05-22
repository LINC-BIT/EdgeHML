#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

import torch
from mmcls.models import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier

from sscl.models import BaseLearningMethod
from .hooks.si_optimizer_hook import SiOptimizerHook
from ssl_utils.models.softteacher.ssod.utils import get_root_logger

import datetime

@CLASSIFIERS.register_module(force=True)
class SiResnet18(ImageClassifier, BaseLearningMethod):
    ALG_NAME = "SiResnet18"  # 算法模型的名称
    ALG_COMPATIBILITY = ["class-il", "task-il", "domain-il"]

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(SiResnet18, self).__init__(
                 backbone,
                 neck=neck,
                 head=head,
                 pretrained=pretrained,
                 train_cfg=train_cfg,
                 init_cfg=init_cfg
        )

        self.big_omega = None
        self.small_omega = 0
        self.c = train_cfg.c
        self.xi = train_cfg.xi
        if not hasattr(self, "device"):
            self.device = "cpu"
        self.checkpoint = self.get_params().data.clone().to(self.device)

        self.learn_timer = 0
        self.learn_timer_per_task = 0
        self.learn_task_cnt = 0

        self.learn_iter_cnt = 0
        self.learn_timer_per_iter = 0
        self.timer100_min = float('inf')

        self.logger = get_root_logger()


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

    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            if self.checkpoint.device != self.device:
                self.checkpoint.to(self.device)
            penalty = (self.big_omega * ((self.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def after_task(self, *args, **kwargs):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.get_params()).to(self.device)

        self.small_omega = self.small_omega.to(self.device)
        self.checkpoint = self.checkpoint.to(self.device)
        self.big_omega += self.small_omega / ((self.get_params().data - self.checkpoint) ** 2 + self.xi)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.get_params().data.clone().to(self.device)
        self.small_omega = 0


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

        self.device = img.device

        losses = super(SiResnet18, self).forward_train(
            img,
            gt_label,
            **kwargs
        )

        penalty = self.penalty()
        losses.update({
            "si_loss": self.c * penalty
        })


        return losses

