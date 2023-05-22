#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

import torch
from mmcls.models import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier

from sscl.utils.buffer import Buffer
from sscl.models import BaseLearningMethod


@CLASSIFIERS.register_module(force=True)
class ErResnet18(ImageClassifier, BaseLearningMethod):
    ALG_NAME = "ErResnet18"  # 算法模型的名称
    ALG_COMPATIBILITY = ["class-il", "task-il", "domain-il"]

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ErResnet18, self).__init__(
                 backbone,
                 neck=neck,
                 head=head,
                 pretrained=pretrained,
                 train_cfg=train_cfg,
                 init_cfg=init_cfg
        )
        self.buffer = Buffer(train_cfg.buffer_size, torch.device("cpu"))
        self.minibatch_size = train_cfg.minibatch_size

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

        losses = super(ErResnet18, self).forward_train(
            img,
            gt_label,
            **kwargs
        )

        buf_data_size = None
        if not self.buffer.is_empty():
            # 目前实现的是往 buffer 中存的就是已经 transform 过的，
            # 而不是从 buffer 中取出来再 transform（后者是 DER 的实现，
            # 可能会性能稍微高一点）
            buf_inputs, buf_labels = self.buffer.get_data(
                self.minibatch_size,
                transform=None
            )
            buf_data_size = buf_labels.shape[0]
            buf_inputs = buf_inputs.to(img.device)
            buf_labels = buf_labels.to(img.device)

            if self.augments is not None:
                buf_inputs, buf_labels = self.augments(buf_inputs, buf_labels)

            x = self.extract_feat(buf_inputs)
            buf_loss = self.head.forward_train(x, buf_labels)
            losses.update({
                f"buf_{k}": v for k, v in buf_loss.items()
            })

        self.buffer.add_data(
            examples=img,
            labels=gt_label
        )

        return losses


if __name__ == "__main__":
    exit(0)
