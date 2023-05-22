#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""
import random

import torch
import numpy as np
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.base_dataset import BaseDataset, expanduser
from mmcls.datasets.pipelines import Compose
from torchvision.datasets import CIFAR100
import copy
import warnings
from ssl_utils.models.softteacher.ssod.utils import get_root_logger
from sscl.datasets.utils.base_task import BaseSplitSemiTask


@DATASETS.register_module(force=True)
class SplitSemiCifar100(BaseSplitSemiTask):

    CLASSES = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
        'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
        'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
        'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
        'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
        'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
        'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
        'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
        'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    # 指定该数据集支持的所有持续学习场景（每次运行仅会使用一种场景设定）：
    DATASET_COMPATIBILITY = ["task-il", "class-il"]

    def __init__(self,
                 root,  # e.g. "data/cifar100"
                 task_id: int,
                 class_index: [list, tuple] = None,  # e.g. [0, 1]
                 num_label_per_class: int = None,  # 5
                 uratio: int = None,  # 7
                 sup_pipeline=None,
                 unsup_weak_pipeline=None,
                 unsup_strong_pipeline=None,
                 test_pipeline=None,
                 test_mode=False,
                 CL_TYPE=None,
                 num_class_per_task: int = 5):
        """
        每个类别的前 num_label_per_class 张图像构成有标签训练集，
        所有图像都构成无标签训练集。

        :param root: 包含 cifar-10-batches-py 的目录路径
        :param class_index: 当前task中需要学习和验证的物体类别
        :param num_label_per_class: 每个类别选几张图像作为有标签图像
        :param uratio: 每一张有标签图像，对应 uratio 张无标签图像，组成一个 batch
        :param sup_pipeline: 有标签图像的pipeline
        :param unsup_weak_pipeline: 无标签图像的弱增强
        :param unsup_strong_pipeline: 无标签图像的强增强
        :param test_mode: 是否是测试模式（而非训练模式）
        """
        super(BaseDataset, self).__init__()

        self.logger = get_root_logger()

        self.test_mode = test_mode

        self.data_prefix = expanduser(root)
        self.sup_pipeline = Compose(sup_pipeline) if not self.test_mode else None
        self.unsup_weak_pipeline = Compose(unsup_weak_pipeline) if not self.test_mode else None
        self.unsup_strong_pipeline = Compose(unsup_strong_pipeline) if not self.test_mode else None
        self.test_pipeline = Compose(test_pipeline) if self.test_mode else None

        self.task_id = task_id
        self.num_class_per_task = num_class_per_task
        if class_index is None:
            self.class_index = [
                i for i in range(
                    self.num_class_per_task * self.task_id,
                    self.num_class_per_task * (self.task_id+1)
                )
            ]
            self.logger.warning(
                f"没有指定当前任务的 class_index，默认该任务包含 {self.num_class_per_task} 个类别，"
                f"将按照 task_id ({self.task_id}) 计算类别范围：{self.class_index}"
            )
        else:
            self.class_index = class_index

        self.num_label_per_class = num_label_per_class
        self.uratio = uratio
        if CL_TYPE is None:
            CL_TYPE = self.DATASET_COMPATIBILITY[0]
            warnings.warn(f"没有指定 task 的 CL_TYPE，将默认使用 {CL_TYPE}")
        assert CL_TYPE in self.DATASET_COMPATIBILITY, f"对 task 指定的场景（{CL_TYPE}）不在数据集兼容范围内（{self.DATASET_COMPATIBILITY}）"
        self.CL_TYPE = CL_TYPE

        self._original_dataset = CIFAR100(root, train=(not test_mode), download=False)

        self.data_infos = self.load_annotations()
        self.unlabeled_data_infos = self.load_unlabeled_infos() if not self.test_mode else None

    def load_annotations(self):
        data_infos = []
        cnt = [0 for _ in range(len(self.CLASSES))]
        for img, gt_label in zip(self._original_dataset.data, self._original_dataset.targets):
            if gt_label in self.class_index:
                if self.test_mode is True or cnt[gt_label] < self.num_label_per_class:
                    cnt[gt_label] += 1
                    data_infos.append({
                        "img": img,
                        "gt_label": np.array(gt_label, dtype=np.int64)
                    })

        # if self.test_mode is True:
        #     data_infos = data_infos[:300]

        cnt = [0 for _ in range(len(self.CLASSES))]
        for d in data_infos:
            cnt[d['gt_label'].item()] += 1
        self.logger.info(f"Task {self.task_id} (test_mode={self.test_mode}) 各类别样本数量：{cnt}")
        return data_infos

    def load_unlabeled_infos(self):
        data_infos = []
        for img, gt_label in zip(self._original_dataset.data, self._original_dataset.targets):
            if gt_label in self.class_index:
                data_infos.append({
                    "img": img,
                    "gt_label": np.array(gt_label, dtype=np.int64),
                })
        return data_infos

    def set_data_infos(self, data_infos):
        self.data_infos = data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.sup_pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        if not self.test_mode:
            data = self.prepare_data(index)
            assert isinstance(data, dict)
            data['task_id'] = self.task_id
            data['index'] = index
            data['unsup'] = {
                'weak': [],
                'strong': []
            }
            # 对每一张有标签图像，随机选取 self.uratio 张无标签图像
            unsup_idx = random.randint(0, len(self.unlabeled_data_infos) - 1)
            for i in range(unsup_idx, unsup_idx+self.uratio):
                idx = i % len(self.unlabeled_data_infos)

                data['unsup']['weak'].append(self.unsup_weak_pipeline(
                    copy.deepcopy(self.unlabeled_data_infos[idx])
                ))
                data['unsup']['weak'][-1]['index'] = idx

                data['unsup']['strong'].append(self.unsup_strong_pipeline(
                    copy.deepcopy(self.unlabeled_data_infos[idx])
                ))
                data['unsup']['strong'][-1]['index'] = idx

            return data

        else:
            results = copy.deepcopy(self.data_infos[index])
            results = self.test_pipeline(results)
            # results['task_id'] = self.task_id
            return results

    @classmethod
    def get_classes(cls, classes=None):
        return cls.CLASSES


if __name__ == "__main__":
    exit(0)
