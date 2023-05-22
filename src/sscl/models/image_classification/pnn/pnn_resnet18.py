#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
EWC Online
"""

from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from mmcls.models import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier

from sscl.models import BaseLearningMethod
from sscl.backbone.ResNet18_PNN import resnet18_pnn
from ssl_utils.models.softteacher.ssod.utils import get_root_logger

import datetime

@CLASSIFIERS.register_module(force=True)
class PnnResnet18(ImageClassifier, BaseLearningMethod):
    ALG_NAME = "PnnResnet18"  # 算法模型的名称
    ALG_COMPATIBILITY = ["task-il"]
    SKIP_TEST_BEFORE_START = True  # 所有task开始前没有net，所以没法测试

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(PnnResnet18, self).__init__(
             backbone,
             neck=neck,
             head=head,
             pretrained=pretrained,
             train_cfg=train_cfg,
             init_cfg=init_cfg
        )

        self.nets = []
        self.net = None
        self.task_id = 0
        self.nf = train_cfg.nf if hasattr(train_cfg, 'nf') else 64

        self.learn_timer = 0
        self.learn_timer_per_task = 0
        self.learn_task_cnt = 0

        self.learn_iter_cnt = 0
        self.learn_timer_per_iter = 0
        self.timer100_min = float('inf')

        self.logger = get_root_logger()

    def to(self, *args, **kwargs):
        if self.net is not None:
            self.net.to(*args, **kwargs)

    def _get_backbone(self, num_classes, nf, x_shape, old_cols=None):
        return resnet18_pnn(num_classes, nf, old_cols, x_shape)

    def before_task(self, dataloaders, runner, *args, **kwargs):
        if runner.task_id >= len(self.nets):
            dl = dataloaders[0]
            num_classes = len(dl.dataset.CLASSES)
            x_shape = dl.dataset.data_infos[0]['img'].shape
            if len(x_shape) == 3:
                assert x_shape[2] in [1, 3], f"通道数应为1或3"
                x_shape = [1, x_shape[2], x_shape[0], x_shape[1]]
            else:
                assert x_shape[1] in [1, 3], f"通道数应为1或3"
            if runner.task_id == 0:
                self.net = self._get_backbone(num_classes, nf=self.nf, x_shape=x_shape)
            else:
                self.net = self._get_backbone(
                    num_classes,
                    nf=self.nf,
                    old_cols=self.nets,  # 这里不能深拷贝，不然self.nets.append(self.net)会导致self.net中的old_cols也被更新
                    x_shape=x_shape
                )
            self.nets.append(self.net)
        else:
            self.net = self.nets[runner.task_id]

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
        self.task_id = runner.task_id

    def before_task_eval(self, runner, *args, **kwargs):
        self.task_id = runner.task_id

    def simple_test(self, img, img_metas=None, **kwargs):
        cls_score = self.nets[self.task_id](img)
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

    def after_task(self, *args, **kwargs):
        pass

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

        logits = self.get_logits(img)
        loss = self.head.loss(logits, gt_label)
        losses = {
            "sup_loss": loss
        }


        return losses
