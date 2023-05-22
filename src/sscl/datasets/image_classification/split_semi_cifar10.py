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
from torchvision.datasets import CIFAR10
import copy
import warnings
from ssl_utils.models.softteacher.ssod.utils import get_root_logger



@DATASETS.register_module(force=True)
class SplitSemiCifar10(BaseDataset):

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]

    # 指定该数据集支持的所有持续学习场景（每次运行仅会使用一种场景设定）：
    DATASET_COMPATIBILITY = ["task-il", "class-il"]

    def __init__(self,
                 root,  # e.g. "data/cifar"
                 task_id: int,
                 class_index: [list, tuple],  # e.g. [0, 1]
                 num_label_per_class: int = None,  # 5
                 uratio: int = None,  # 7
                 sup_pipeline=None,
                 unsup_weak_pipeline=None,
                 unsup_strong_pipeline=None,
                 test_pipeline=None,
                 test_mode=False,
                 CL_TYPE=None,
                 unsup_sample_mode=1):
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

        self.unsup_sample_mode = unsup_sample_mode
        self.logger.info(f"当前无标注样本采样模式：{self.unsup_sample_mode}")

        self.data_prefix = expanduser(root)
        self.sup_pipeline = Compose(sup_pipeline) if not self.test_mode else None
        self.unsup_weak_pipeline = Compose(unsup_weak_pipeline) if not self.test_mode else None
        self.unsup_strong_pipeline = Compose(unsup_strong_pipeline) if not self.test_mode else None
        self.test_pipeline = Compose(test_pipeline) if self.test_mode else None

        self.task_id = task_id
        self.class_index = class_index
        self.num_label_per_class = num_label_per_class
        self.uratio = uratio
        if CL_TYPE is None:
            CL_TYPE = self.DATASET_COMPATIBILITY[0]
            warnings.warn(f"没有指定 task 的 CL_TYPE，将默认使用 {CL_TYPE}")
        assert CL_TYPE in self.DATASET_COMPATIBILITY, f"对 task 指定的场景（{CL_TYPE}）不在数据集兼容范围内（{self.DATASET_COMPATIBILITY}）"
        self.CL_TYPE = CL_TYPE

        self._original_dataset = CIFAR10(root, train=(not test_mode), download=False)

        self.data_infos = self.load_annotations()
        self.unlabeled_data_infos = self.load_unlabeled_infos() if not self.test_mode else None

        self.get_train_sample = 0

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

        # for debugging
        # if self.test_mode is True:
        #     data_infos = data_infos[:30]

        cnt = [0 for _ in range(len(self.CLASSES))]
        for d in data_infos:
            cnt[d['gt_label'].item()] += 1
        self.logger.info(f"Task {self.task_id} (test_mode={self.test_mode}) 各类别样本数量：{cnt}")
        return data_infos

    def load_annotations_rand(self):
        data_infos = []

        if self.test_mode is True:
            for img, gt_label in zip(self._original_dataset.data, self._original_dataset.targets):
                if gt_label in self.class_index:
                    data_infos.append({
                        "img": img,
                        "gt_label": np.array(gt_label, dtype=np.int64)
                    })
        else:
            self.logger.info(f"开始随机选取 {self.num_label_per_class}/class 个标注样本")
            indexes_of_each_class = {
                c: [] for c in self.class_index
            }
            # 随机选取 self.num_label_per_class 张标注
            all_samples = list(zip(self._original_dataset.data, self._original_dataset.targets))
            for idx, (img, gt_label) in enumerate(all_samples):
                if gt_label in self.class_index:
                    indexes_of_each_class[gt_label].append(idx)
            for cls in self.class_index:
                selected_indexes = np.random.choice(
                    indexes_of_each_class[cls],
                    size=self.num_label_per_class
                )
                self.logger.info(f"类别 [{cls}] 选取标注样本索引：{selected_indexes}")
                for idx in selected_indexes:
                    data_infos.append({
                        "img": all_samples[idx][0],
                        "gt_label": np.array(all_samples[idx][1], dtype=np.int64)
                    })
            assert len(data_infos) == self.num_label_per_class * len(self.class_index)

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
                    "gt_label": np.array(gt_label, dtype=np.int64),  # 无标注样本也放入标注，但不应该使用
                })
        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.sup_pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    """
    注意，当前版本的数据集有bug，即 __len__() 只返回了有标注样本数量（10个），
    这导致 pytorch 会默认 1 个 epoch 是只有 10 个样本，所以会进而导致多个问题：
    1. mmcv 会在 1 个 epoch 之后 sleep(2s) 来避免某个死锁，因此频繁的 epoch 会
        导致大量时间都花在 sleep 上了；
    2. pytorch 的 dataloader 中，persistent_workers 设为 False 的话，会在每个
        epoch 完成后销毁 worker，下一个 epoch 再创建新的 worker，这会导致每个
        epoch 完成之后，这个 SplitSemiCifar10 对象都会被创建一遍，也因此下面的
        __getitem__() 中取随机无标注样本索引的“随机过程”会一直重复，因为随机种子
        一直是 0，所以相当于每个 epoch 取到的无标注样本都是一样的。（有标注样本不一样，
        因为有标注样本的 index 是主线程随机的，主线程没有被销毁）。
    
    与此同时，workers_per_gpu 最好设为 1，防止多个 worker 由于随机种子一样进而
    导致每次随机出的无标注样本索引也一样
    """

    def __getitem__(self, index):
        if not self.test_mode:
            # self.get_train_sample += 1
            # if self.get_train_sample % 100 == 0:
            #     self.logger.info(f"get_train_sample = {self.get_train_sample}")
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
