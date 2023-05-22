#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

import torch
from mmcls.models import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier

from sscl.utils.buffer import Buffer
from sscl.models import BaseLearningMethod
from .er_resnet18 import ErResnet18
from ssl_utils.models.fixmatch.fixmatch_utils import consistency_loss


@CLASSIFIERS.register_module(force=True)
class ErFixmatchResnet18(ErResnet18):
    ALG_NAME = "ErFixmatchResnet18"  # 算法模型的名称
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
        self.p_cutoff = 0.95  # 置信度高于τ的prob（prob[i] = softmax(logits)[i]）


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

        losses = super(ErFixmatchResnet18, self).forward_train(
            img,
            gt_label,
            **kwargs
        )

        # FixMatch无监督loss：
        if self.lambda_u > 0:
            weak_data = self.extract_unsup_cls_data(kwargs['unsup']['weak'])
            strong_data = self.extract_unsup_cls_data(kwargs['unsup']['strong'])

            logits_weak = self.get_logits(weak_data['img'])
            logits_strong = self.get_logits(strong_data['img'])
            unsup_loss, mask, select, pseudo_lb = consistency_loss(
                logits_strong,  # 无标注数据、强增强的logits
                logits_weak,  # 无标注数据，弱增强的logits
                'ce', self.T, self.p_cutoff,
                use_hard_labels=True)
            losses['unsup_loss'] = self.lambda_u * unsup_loss

        return losses

    def get_logits(self, img):
        x = self.extract_feat(img)
        x = self.head.pre_logits(x)
        cls_score = self.head.fc(x)
        return cls_score
