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
import copy
import warnings
from ssl_utils.models.softteacher.ssod.utils import get_root_logger
import os
from pathlib import Path
import cv2


@DATASETS.register_module(force=True)
class SplitSemiTinyimagenet(BaseDataset):

    CLASSES = [
        "n01443537",
        "n01629819",
        "n01641577",
        "n01644900",
        "n01698640",
        "n01742172",
        "n01768244",
        "n01770393",
        "n01774384",
        "n01774750",
        "n01784675",
        "n01855672",
        "n01882714",
        "n01910747",
        "n01917289",
        "n01944390",
        "n01945685",
        "n01950731",
        "n01983481",
        "n01984695",
        "n02002724",
        "n02056570",
        "n02058221",
        "n02074367",
        "n02085620",
        "n02094433",
        "n02099601",
        "n02099712",
        "n02106662",
        "n02113799",
        "n02123045",
        "n02123394",
        "n02124075",
        "n02125311",
        "n02129165",
        "n02132136",
        "n02165456",
        "n02190166",
        "n02206856",
        "n02226429",
        "n02231487",
        "n02233338",
        "n02236044",
        "n02268443",
        "n02279972",
        "n02281406",
        "n02321529",
        "n02364673",
        "n02395406",
        "n02403003",
        "n02410509",
        "n02415577",
        "n02423022",
        "n02437312",
        "n02480495",
        "n02481823",
        "n02486410",
        "n02504458",
        "n02509815",
        "n02666196",
        "n02669723",
        "n02699494",
        "n02730930",
        "n02769748",
        "n02788148",
        "n02791270",
        "n02793495",
        "n02795169",
        "n02802426",
        "n02808440",
        "n02814533",
        "n02814860",
        "n02815834",
        "n02823428",
        "n02837789",
        "n02841315",
        "n02843684",
        "n02883205",
        "n02892201",
        "n02906734",
        "n02909870",
        "n02917067",
        "n02927161",
        "n02948072",
        "n02950826",
        "n02963159",
        "n02977058",
        "n02988304",
        "n02999410",
        "n03014705",
        "n03026506",
        "n03042490",
        "n03085013",
        "n03089624",
        "n03100240",
        "n03126707",
        "n03160309",
        "n03179701",
        "n03201208",
        "n03250847",
        "n03255030",
        "n03355925",
        "n03388043",
        "n03393912",
        "n03400231",
        "n03404251",
        "n03424325",
        "n03444034",
        "n03447447",
        "n03544143",
        "n03584254",
        "n03599486",
        "n03617480",
        "n03637318",
        "n03649909",
        "n03662601",
        "n03670208",
        "n03706229",
        "n03733131",
        "n03763968",
        "n03770439",
        "n03796401",
        "n03804744",
        "n03814639",
        "n03837869",
        "n03838899",
        "n03854065",
        "n03891332",
        "n03902125",
        "n03930313",
        "n03937543",
        "n03970156",
        "n03976657",
        "n03977966",
        "n03980874",
        "n03983396",
        "n03992509",
        "n04008634",
        "n04023962",
        "n04067472",
        "n04070727",
        "n04074963",
        "n04099969",
        "n04118538",
        "n04133789",
        "n04146614",
        "n04149813",
        "n04179913",
        "n04251144",
        "n04254777",
        "n04259630",
        "n04265275",
        "n04275548",
        "n04285008",
        "n04311004",
        "n04328186",
        "n04356056",
        "n04366367",
        "n04371430",
        "n04376876",
        "n04398044",
        "n04399382",
        "n04417672",
        "n04456115",
        "n04465501",
        "n04486054",
        "n04487081",
        "n04501370",
        "n04507155",
        "n04532106",
        "n04532670",
        "n04540053",
        "n04560804",
        "n04562935",
        "n04596742",
        "n04597913",
        "n06596364",
        "n07579787",
        "n07583066",
        "n07614500",
        "n07615774",
        "n07695742",
        "n07711569",
        "n07715103",
        "n07720875",
        "n07734744",
        "n07747607",
        "n07749582",
        "n07753592",
        "n07768694",
        "n07871810",
        "n07873807",
        "n07875152",
        "n07920052",
        "n09193705",
        "n09246464",
        "n09256479",
        "n09332890",
        "n09428293",
        "n12267677",
    ]

    # 指定该数据集支持的所有持续学习场景（每次运行仅会使用一种场景设定）：
    DATASET_COMPATIBILITY = ["task-il", "class-il"]

    def __init__(self,
                 root,  # e.g. "data/cifar"
                 task_id: int,
                 class_index: [list, tuple, dict],  # e.g. [0, 1]
                 num_label_per_class: int = None,  # 5
                 uratio: int = None,  # 7
                 sup_pipeline=None,
                 unsup_weak_pipeline=None,
                 unsup_strong_pipeline=None,
                 test_pipeline=None,
                 test_mode=False,
                 CL_TYPE=None):
        """
        每个类别的前 num_label_per_class 张图像构成有标签训练集，
        所有图像都构成无标签训练集。

        :param root: 包含 train、val、test 文件夹的目录路径；
                        下载地址：https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
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
        if test_mode is True:
            self.data_prefix = os.path.join(self.data_prefix, "val")
        else:
            self.data_prefix = os.path.join(self.data_prefix, "train")

        self.sup_pipeline = Compose(sup_pipeline) if not self.test_mode else None
        self.unsup_weak_pipeline = Compose(unsup_weak_pipeline) if not self.test_mode else None
        self.unsup_strong_pipeline = Compose(unsup_strong_pipeline) if not self.test_mode else None
        self.test_pipeline = Compose(test_pipeline) if self.test_mode else None

        self.task_id = task_id
        if isinstance(class_index, (list, tuple)):
            self.class_index = class_index
        else:
            self.class_index = [i for i in range(class_index['begin'], class_index['end'])]

        self.num_label_per_class = num_label_per_class
        self.uratio = uratio
        if CL_TYPE is None:
            CL_TYPE = self.DATASET_COMPATIBILITY[0]
            warnings.warn(f"没有指定 task 的 CL_TYPE，将默认使用 {CL_TYPE}")
        assert CL_TYPE in self.DATASET_COMPATIBILITY, f"对 task 指定的场景（{CL_TYPE}）不在数据集兼容范围内（{self.DATASET_COMPATIBILITY}）"
        self.CL_TYPE = CL_TYPE

        # self._original_dataset = CIFAR10(root, train=(not test_mode), download=False)

        if test_mode is True:
            self.all_labels = self.load_val_labels()

        self.data_infos = self.load_annotations()
        self.unlabeled_data_infos = self.load_unlabeled_infos() if not self.test_mode else None

    def load_val_labels(self):
        """
        读取验证集的标注文件
        :return:
        """
        val_path = Path(self.data_prefix) / "val_annotations.txt"
        class_names = [self.CLASSES[i] for i in self.class_index]
        labels = {}
        with open(str(val_path), 'r') as file:
            for line in file.readlines():
                line_infos = line.split('\t')
                img_file_name = line_infos[0]
                img_cls_name = line_infos[1]
                if img_cls_name in class_names:
                    labels[img_file_name] = img_cls_name
        return labels

    def load_annotations(self):
        data_infos = []
        cnt = [0 for _ in range(len(self.CLASSES))]
        img_root_dir = Path(self.data_prefix)
        if self.test_mode is False:
            # 训练数据：
            for cls in self.class_index:
                class_name = self.CLASSES[cls]
                class_img_dir = img_root_dir / class_name / "images"
                for img_path in class_img_dir.iterdir():
                    img = cv2.imread(str(img_path))
                    if self.test_mode is True or cnt[cls] < self.num_label_per_class:
                        cnt[cls] += 1
                        data_infos.append({
                            "img": img,
                            "gt_label": np.array(cls, dtype=np.int64)
                        })
        else:
            # 验证集
            img_dir = img_root_dir / "images"
            for img_file_name, img_cls_name in self.all_labels.items():
                img = cv2.imread(str(img_dir / img_file_name))
                cls_index = self.CLASSES.index(img_cls_name)
                cnt[cls_index] += 1
                data_infos.append({
                    "img": img,
                    "gt_label": np.array(
                        cls_index,
                        dtype=np.int64
                    )
                })

        cnt = [0 for _ in range(len(self.CLASSES))]
        for d in data_infos:
            cnt[d['gt_label'].item()] += 1
        self.logger.info(f"Task {self.task_id} (test_mode={self.test_mode}) 各类别样本数量：{cnt}")
        return data_infos

    def load_unlabeled_infos(self):
        data_infos = []
        img_root_dir = Path(self.data_prefix)
        for cls in self.class_index:
            class_name = self.CLASSES[cls]
            class_img_dir = img_root_dir / class_name / "images"
            for img_path in class_img_dir.iterdir():
                img = cv2.imread(str(img_path))
                data_infos.append({
                    "img": img,
                    "gt_label": None
                })

        return data_infos

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
            return results

    @classmethod
    def get_classes(cls, classes=None):
        return cls.CLASSES


if __name__ == "__main__":
    exit(0)
