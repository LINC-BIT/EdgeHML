#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

from mmcv.cnn import ConvModule
from mmcls.models import BACKBONES, MobileNetV2
from mmcls.models.utils import make_divisible


# ==============================================================
# 以下对于 MMClassification 中 MobileNetV2 进行调整，尤其是调整了 stride，
# 如果不调整的话，loss会爆、准确率难以上升。
# 参考：sscl/backbone/image_classification_pytorch/mobilenet.py

@BACKBONES.register_module(force=True)
class MobileNetV2_CIFAR(MobileNetV2):
    arch_settings = [
        [1, 16, 1, 1],
        [6, 24, 2, 1],  # 最后一项stride由2改为1
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]]

    def __init__(self,
                 widen_factor=1.0,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):

        super(MobileNetV2_CIFAR, self).__init__(
                 widen_factor=widen_factor,
                 out_indices=out_indices,
                 frozen_stages=frozen_stages,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 act_cfg=act_cfg,
                 norm_eval=norm_eval,
                 with_cp=with_cp,
                 init_cfg=init_cfg)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=make_divisible(32 * widen_factor, 8),
            kernel_size=3,
            stride=1,  # stride由2改为1
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
