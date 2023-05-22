#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""
import torch
from mmcls.models import CLASSIFIERS
from ssl_utils.models.uda.uda_utils import consistency_loss, TSA
from ssl_utils.train_utils import ce_loss
from .derpp_resnet18 import DerppResnet18

import datetime

@CLASSIFIERS.register_module(force=True)
class DerppUdaResnet18(DerppResnet18):
    ALG_NAME = "DerppUdaResnet18"  # 算法模型的名称
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
        self.T = 0.4
        self.p_cutoff = 0.8  # 置信度高于τ的prob（prob[i] = softmax(logits)[i]）
        self.TSA_schedule = 'none'
        self.classwise_acc = None
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
        self.classwise_acc = torch.zeros((n_classes,))
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

        logits = self.get_logits(img)
        losses = self.derpp_forward_train(img, gt_label, logits, **kwargs)

        # UDA对有监督loss也有特殊处理：
        inputs = img
        labels = gt_label
        logits_x_lb = self.get_logits(inputs)
        tsa = TSA(
            self.TSA_schedule, self.iter_cnt, self.num_iters,
            self.n_classes
        )  # Training Signal Annealing
        self.iter_cnt += 1

        sup_mask = torch.max(torch.softmax(logits_x_lb, dim=-1), dim=-1)[0].le(tsa).float().detach()
        sup_loss = (ce_loss(logits_x_lb, labels, reduction='none') * sup_mask).mean()
        losses.update({
            "sup_loss": sup_loss,
        })

        # UDA无监督loss：
        if self.lambda_u > 0:
            weak_data = self.extract_unsup_cls_data(kwargs['unsup']['weak'])
            strong_data = self.extract_unsup_cls_data(kwargs['unsup']['strong'])

            unsup_loss, mask, select, pseudo_lb = consistency_loss(
                self.get_logits(strong_data['img']),
                self.get_logits(weak_data['img']),
                self.classwise_acc,
                self.iter_cnt,
                None,  # 这个参数其实没用
                'ce', self.T, self.p_cutoff,
                use_flex=False)

            # 这两个跟Flex有关，在这里不发挥作用：
            # if x_ulb_idx[select == 1].nelement() != 0:
            #     selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

            losses.update({
                "unsup_loss": self.lambda_u * unsup_loss
            })


        return losses

