#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

from mmcls.models import NECKS
from torch import nn
import torch

@NECKS.register_module()
class AvgPool2d(nn.Module):
    """
    因为 MMClassification 没有实现 pytorch 中原生的 AvgPool2d，所以特此实现
    """

    def __init__(self, *args, **kwargs):
        super(AvgPool2d, self).__init__(*args, **kwargs)
        self.avg_pool = None

    def forward(self, inputs):
        if self.avg_pool is None:
            if isinstance(inputs, tuple):
                size = (inputs[0].size(-2), inputs[0].size(-1))
            elif isinstance(inputs, torch.Tensor):
                size = (inputs.size(-2), inputs.size(-1))
            else:
                raise TypeError('neck inputs should be tuple or torch.tensor')
            self.avg_pool = nn.AvgPool2d(size)

        if isinstance(inputs, tuple):
            outs = tuple([
                self.avg_pool(x)
                for x in inputs
            ])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.avg_pool(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
