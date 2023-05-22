#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

from mmcls.models.backbones import ResNet_CIFAR
from mmcls.models.builder import BACKBONES


@BACKBONES.register_module(force=True)
class ResNet_CIFAR_Freeze(ResNet_CIFAR):

    def clear_grad(self, num_stage: int=4):
        if num_stage >= 0:
            if self.deep_stem:
                for param in self.stem.parameters():
                    param.grad.mul_(0)
            else:
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.grad.mul_(0)

        for i in range(1, num_stage + 1):
            m = getattr(self, f'layer{i}')
            for param in m.parameters():
                param.grad.mul_(0)

    def freeze_backbone(self):
        self.frozen_stages = 4
        self._freeze_stages()

    def unfreeze_backbone(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.train()
                for param in self.stem.parameters():
                    param.requires_grad = True
            else:
                for m in [self.conv1]:
                    for param in m.parameters():
                        param.requires_grad = True
                if self.norm_eval is False:
                    self.norm1.train()
                    for m in [self.norm1]:
                        for param in m.parameters():
                            param.requires_grad = True

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.train()
            for param in m.parameters():
                param.requires_grad = True

        self.frozen_stages = -1

