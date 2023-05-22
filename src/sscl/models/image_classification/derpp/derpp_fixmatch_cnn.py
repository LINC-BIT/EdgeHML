#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
这个文件包括了除了Resnet18以外的常用CNN
"""

from copy import deepcopy
import torch.nn.functional as F
from mmcls.models import CLASSIFIERS

from .derpp_fixmatch_resnet18 import DerppFixmatchResnet18

# ==============================================================
# 这些类直接使用 MMClassification 中的 CNN（或基于其进行进行调整）作为 backbone：

@CLASSIFIERS.register_module(force=True)
class DerppFixmatchMobilenetv2(DerppFixmatchResnet18):
    ALG_NAME = "DerppFixmatchMobilenetv2"

    from sscl.backbone.image_classification import MobileNetV2_CIFAR


@CLASSIFIERS.register_module(force=True)
class DerppFixmatchWideResnet50(DerppFixmatchResnet18):
    ALG_NAME = "DerppFixmatchWideResnet50"


# 会爆loss：
# @CLASSIFIERS.register_module(force=True)
# class DerppFixmatchResnext50(DerppFixmatchResnet18):
#     ALG_NAME = "DerppFixmatchResnext50"

# ==============================================================
# 以下是完全使用 sscl/backbone/image_classification_pytorch 中的 CNN 作为整个网络，
# 对于 MobileNetV2来说，会比上面的准确率稍高点，可能因为具体 CNN 中的 block 稍有差异：

@CLASSIFIERS.register_module(force=True)
class DerppFixmatchMobilenetv2_v2(DerppFixmatchResnet18):
    ALG_NAME = "DerppFixmatchMobilenetv2_v2"
    SKIP_TEST_BEFORE_START = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.net = None

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.mobilenet import mobilenetv2
        return mobilenetv2(num_classes)

    def to(self, *args, **kwargs):
        if self.net is not None:
            self.net.to(*args, **kwargs)

    def before_task(self, dataloaders, runner, *args, **kwargs):
        if self.net is None:
            dl = dataloaders[0]
            num_classes = len(dl.dataset.CLASSES)
            self.net = self.get_backbone(num_classes)

            self.net.to(runner.optimizer.param_groups[0]['params'][0].device)
            # 直接删除optimizer中的原param_group然后添加新的param_group会报错，
            # 因为optimizer内部还存储有原来param_group的参数，所以会导致optimzer.step
            # 时找不到原param_group。所以只能重新创建optimizer再绑定到runner上了：
            obj_cls = type(runner.optimizer)
            optimizer_config = deepcopy(runner.optimizer.defaults)
            optimizer_config['params'] = self.net.parameters()
            del runner.optimizer
            runner.optimizer = obj_cls(**optimizer_config)
            print(f"已更换 optimizer：对新的 net 使用 defaults config 构造了 optimizer")

    def simple_test(self, img, img_metas=None, **kwargs):
        cls_score = self.net(img)
        softmax = kwargs['softmax'] if 'softmax' in kwargs else True
        post_process = kwargs['post_process'] if 'post_process' in kwargs else True

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.head.post_process(pred)
        else:
            return pred

    def get_logits(self, img):
        x = self.net(img)
        return x


@CLASSIFIERS.register_module(force=True)
class DerppFixmatchRAN_v2(DerppFixmatchMobilenetv2_v2):
    ALG_NAME = "DerppFixmatchRAN_v2"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.residual_attention_network import ResidualAttentionModel_92_32input_update as RAN
        return RAN(num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppFixmatchCBAM_v2(DerppFixmatchMobilenetv2_v2):
    ALG_NAME = "DerppFixmatchCBAM_v2"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.cbam import cbamnet
        return cbamnet(network_type='CIFAR100', num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppFixmatchResnext29_v2(DerppFixmatchMobilenetv2_v2):
    ALG_NAME = "DerppFixmatchResnext29_v2"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.resnext import resnext29_2x64d
        return resnext29_2x64d(num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppFixmatchInceptionv3_v2(DerppFixmatchMobilenetv2_v2):
    ALG_NAME = "DerppFixmatchInceptionv3_v2"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.inception import inceptionv3
        return inceptionv3(num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppFixmatchSeresnet18_v2(DerppFixmatchMobilenetv2_v2):
    ALG_NAME = "DerppFixmatchSeresnet18_v2"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.senet import senet18
        return senet18(num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppFixmatchVGG16_v2(DerppFixmatchMobilenetv2_v2):
    ALG_NAME = "DerppFixmatchVGG16_v2"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.vgg import vgg16
        return vgg16(num_classes=num_classes)

    # 对于neck=None、head为ClsHead的model，需要使用类似于Lenet的前馈计算：
    # def get_logits(self, img):
    #     cls_score = self.extract_feat(img)
    #     if isinstance(cls_score, tuple):
    #         cls_score = cls_score[-1]
    #     return cls_score
