#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""
import torch
from mmcls.models import CLASSIFIERS
from copy import deepcopy
from collections import Counter

from ssl_utils.models.flexmatch.flexmatch_utils import consistency_loss
from .derpp_resnet18 import DerppResnet18

import datetime

@CLASSIFIERS.register_module(force=True)
class DerppFlexmatchResnet18(DerppResnet18):
    ALG_NAME = "DerppFlexmatchResnet18"  # 算法模型的名称
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
        self.hard_label = True
        self.use_DA = False
        self.thresh_warmup = True

        self.p_target = None
        self.p_model = None
        self.n_classes = None
        self.n_ulb_dset = None
        self.selected_label = None
        self.classwise_acc = None

    def before_task(self, dataloaders, *args, **kwargs):
        dataset = dataloaders[0].dataset
        if "class-il" in dataset.DATASET_COMPATIBILITY or "domain-il" in dataset.DATASET_COMPATIBILITY:
            n_classes = len(dataset.CLASSES)
        else:
            n_classes = len(dataset.class_index)

        cnts = [0 for _ in range(n_classes)]
        for info in dataset.data_infos:
            cnts[info['gt_label'].item()] += 1
        cnts = [x / sum(cnts) for x in cnts]
        label_distribution = torch.tensor(cnts, dtype=torch.float32)

        self.p_target = label_distribution
        self.p_model = None
        self.n_classes = n_classes
        self.n_ulb_dset = len(dataset.unlabeled_data_infos)

        self.selected_label = torch.ones((self.n_ulb_dset,), dtype=torch.long, ) * -1
        self.classwise_acc = torch.zeros((n_classes,))

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

        losses, logits = self.sup_forward_train(img, gt_label, **kwargs)
        losses_derpp = self.derpp_forward_train(img, gt_label, logits, **kwargs)
        losses.update(losses_derpp)

        # FlexMatch无监督loss：
        if self.lambda_u > 0:

            weak_data = self.extract_unsup_cls_data(kwargs['unsup']['weak'])
            strong_data = self.extract_unsup_cls_data(kwargs['unsup']['strong'])
            x_ulb_w = weak_data['img']
            x_ulb_s = strong_data['img']

            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]
            device = x_ulb_w.device
            x_ulb_idx = torch.stack(weak_data['index'])
            x_ulb_idx = x_ulb_idx.to(device)
            self.selected_label = self.selected_label.to(device)
            self.classwise_acc = self.classwise_acc.to(device)

            pseudo_counter = Counter(self.selected_label.tolist())
            if max(pseudo_counter.values()) < self.n_ulb_dset:  # not all(5w) -1
                if self.thresh_warmup:
                    for i in range(self.n_classes):
                        self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(self.n_classes):
                        self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            inputs = torch.cat((x_ulb_w, x_ulb_s))
            logits = self.get_logits(inputs)
            logits_x_ulb_w, logits_x_ulb_s = logits.chunk(2)

            # hyper-params for update
            T = self.T
            p_cutoff = self.p_cutoff
            unsup_loss, mask, select, pseudo_lb, p_model = consistency_loss(
                logits_x_ulb_s,
                logits_x_ulb_w,
                self.classwise_acc,
                self.p_target,
                self.p_model,
                'ce', T, p_cutoff,
                use_hard_labels=self.hard_label,
                use_DA=self.use_DA)

            if x_ulb_idx[select == 1].nelement() != 0:
                self.selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

            losses['unsup_loss'] = self.lambda_u * unsup_loss


        return losses

    def extract_unsup_cls_data(self, raw_data: [list, tuple]):
        all_img_metas = []
        all_img = None
        all_img_index = []
        for d in raw_data:
            all_img_metas.extend(d['img_metas'])
            if "index" in d:
                all_img_index.extend(d['index'])
            if all_img is None:
                all_img = d['img']
            else:
                all_img = torch.cat([all_img, d['img']])
        return {
            'img_metas': all_img_metas,
            'img': all_img,
            'index': all_img_index
        }

