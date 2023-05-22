#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

import torch
from torch.nn import functional as F
from mmcls.models import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier

from sscl.utils.buffer import Buffer
from sscl.models import BaseLearningMethod


@CLASSIFIERS.register_module(force=True)
class DerppLenet(ImageClassifier, BaseLearningMethod):
    ALG_NAME = "DerppLenet"  # 算法模型的名称
    ALG_COMPATIBILITY = ["class-il", "task-il", "domain-il"]

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(DerppLenet, self).__init__(
                 backbone,
                 neck=neck,
                 head=head,
                 pretrained=pretrained,
                 train_cfg=train_cfg,
                 init_cfg=init_cfg
        )
        self.buffer = Buffer(train_cfg.buffer_size, torch.device("cpu"))
        self.minibatch_size = train_cfg.minibatch_size

        self.alpha = train_cfg.alpha
        self.beta = train_cfg.beta

    def get_logits(self, img):
        cls_score = self.extract_feat(img)
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        return cls_score

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

    def derpp_forward_train(self, img, gt_label, logits: torch.Tensor, **kwargs):
        loss = {}

        assert self.alpha >= 0 and self.beta >= 0

        # buffer中的data replay：
        if not self.buffer.is_empty():  # 从buffer中选取logits和labels用于replay
            if self.alpha > 0:
                buf_inputs, _, buf_logits = self.buffer.get_data(
                    self.minibatch_size,
                    transform=None
                )
                buf_inputs = buf_inputs.to(img.device)
                buf_logits = buf_logits.to(img.device)

                buf_outputs = self.get_logits(buf_inputs)

                alpha_loss = self.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss['alpha_loss'] = alpha_loss

            if self.beta > 0:
                buf_inputs, buf_labels, _ = self.buffer.get_data(
                    self.minibatch_size,
                    transform=None
                )
                buf_inputs = buf_inputs.to(img.device)
                buf_labels = buf_labels.to(img.device)

                buf_outputs = self.get_logits(buf_inputs)

                beta_loss = self.beta * F.cross_entropy(buf_outputs, buf_labels)
                loss['beta_loss'] = beta_loss

        if self.alpha > 0 or self.beta > 0:
            logits_leaved_graph = logits.detach()
            self.buffer.add_data(
                examples=img,  # 在原DERPP官方实现中，这里保存的应该是没有被transform的img
                labels=gt_label,
                logits=logits_leaved_graph
            )

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

        losses = {}
        losses_sup, logits = self.sup_forward_train(img, gt_label, **kwargs)
        losses.update(losses_sup)
        losses_derpp = self.derpp_forward_train(img, gt_label, logits, **kwargs)
        losses.update(losses_derpp)

        return losses

