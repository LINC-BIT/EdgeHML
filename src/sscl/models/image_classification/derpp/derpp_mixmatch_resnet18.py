#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

import torch
import numpy as np

from mmcls.models import CLASSIFIERS
from ssl_utils.train_utils import ce_loss, Bn_Controller
from ssl_utils.models.mixmatch.mixmatch_utils import consistency_loss, one_hot, mixup_one_target, MixMatchTransform
from .derpp_resnet18 import DerppResnet18


@CLASSIFIERS.register_module(force=True)
class DerppMixmatchResnet18(DerppResnet18):
    ALG_NAME = "DerppMixmatchResnet18"  # 算法模型的名称
    ALG_COMPATIBILITY = ["class-il", "task-il", "domain-il"]

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(
             backbone,
             neck=neck,
             head=head,
             pretrained=pretrained,
             train_cfg=train_cfg,
             init_cfg=init_cfg
        )

        self.lambda_u = train_cfg.lambda_u
        self.T = 0.5
        self.mixmatch_alpha = 0.5  # 要与derpp的alpha作区分
        self.ramp_up = 0.4
        self.bn_controller = Bn_Controller()
        self.n_classes = None
        self.num_iters = 0  # 每个task上的迭代次数
        self.iter_cnt = 0  # 计数器，每次迭代+1

    def before_task(self, dataloaders, num_iters, *args, **kwargs):
        dataset = dataloaders[0].dataset
        if "class-il" in dataset.DATASET_COMPATIBILITY or "domain-il" in dataset.DATASET_COMPATIBILITY:
            n_classes = len(dataset.CLASSES)
        else:
            n_classes = len(dataset.class_index)

        self.n_classes = n_classes
        self.num_iters = num_iters
        self.iter_cnt = 0

    def forward_train(self, img, gt_label, **kwargs):
        """

        :param img: (N, C, H, W)
        :param gt_label: (N, 1)
        :param kwargs: 包含了 img_metas 和 unsup 样本数据
                kwargs['unsup']['weak'][0] = {
                    'img_metas': [img_meta_dict, img_meta_dict, ...],
                    'img': N*C*H*W 的 tensor
        :return:
        """

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        derpp_logits = self.get_logits(img)
        losses = self.derpp_forward_train(img, gt_label, derpp_logits, **kwargs)

        weak_data = self.extract_unsup_cls_data(kwargs['unsup']['weak'])
        strong_data = self.extract_unsup_cls_data(kwargs['unsup']['strong'])

        x_ulb_w1 = weak_data['img']
        x_ulb_w2 = strong_data['img']  # MixMatch这里是用了2次弱增强得到了w1和w2
        x_lb = img
        y_lb = gt_label

        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w1.shape[0]
        assert num_ulb == x_ulb_w2.shape[0]

        with torch.no_grad():
            self.bn_controller.freeze_bn(self)
            logits_x_ulb_w1 = self.get_logits(x_ulb_w1)
            logits_x_ulb_w2 = self.get_logits(x_ulb_w2)
            self.bn_controller.unfreeze_bn(self)
            # Temperature sharpening
            T = self.T
            # avg
            avg_prob_x_ulb = (torch.softmax(logits_x_ulb_w1, dim=1) + torch.softmax(logits_x_ulb_w2, dim=1)) / 2
            avg_prob_x_ulb = (avg_prob_x_ulb / avg_prob_x_ulb.sum(dim=-1, keepdim=True))
            # sharpening
            sharpen_prob_x_ulb = avg_prob_x_ulb ** (1 / T)
            sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

            # Pseudo Label
            input_labels = torch.cat(
                [
                    one_hot(
                        y_lb,
                        self.n_classes,
                        x_lb.device,
                    ),
                    sharpen_prob_x_ulb,
                    sharpen_prob_x_ulb
                ],
                dim=0
            )

            # Mix up
            inputs = torch.cat([x_lb, x_ulb_w1, x_ulb_w2])
            mixed_x, mixed_y, _ = mixup_one_target(
                inputs,
                input_labels,
                x_lb.device,
                self.mixmatch_alpha,
                is_bias=True)

            # Interleave labeled and unlabeled samples between batches to get correct batch norm calculation
            mixed_x = list(torch.split(mixed_x, num_lb))
            mixed_x = self.interleave(mixed_x, num_lb)

        logits = [self.get_logits(mixed_x[0])]
        # calculate BN for only the first batch
        self.bn_controller.freeze_bn(self)
        for ipt in mixed_x[1:]:
            logits.append(self.get_logits(ipt))

        # put interleaved samples back
        logits = self.interleave(logits, num_lb)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        self.bn_controller.unfreeze_bn(self)

        sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False)
        losses.update({
            "sup_loss": sup_loss.mean()
        })

        unsup_loss = consistency_loss(logits_u, mixed_y[num_lb:])

        # set ramp_up for lambda_u
        rampup = float(np.clip(self.iter_cnt / (self.ramp_up * self.num_iters), 0.0, 1.0))
        self.iter_cnt += 1
        lambda_u = self.lambda_u * rampup

        losses.update({
            "unsup_loss": lambda_u * unsup_loss
        })

        return losses

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

