#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

from torchvision.datasets import CIFAR10
import cv2
from pathlib import Path

def read_cifar10(root="data/cifar", mode="train", num_task=5, num_label_per_class=5):
    num_class = 10
    num_class_per_task = num_class // num_task
    dataset = CIFAR10(
        root=root,
        train=(mode == "train"),
        download=False
    )
    root_dir = Path(root) / "cifar10-sscl" / mode
    img_dir = root_dir / "images"
    ann_dir = root_dir / "annotations"
    if not img_dir.exists() or not img_dir.is_dir():
        img_dir.mkdir(parents=True)
    if not ann_dir.exists() or not ann_dir.is_dir():
        ann_dir.mkdir(parents=True)

    anns = [
        {
            "sup": [],
            "unsup": [],
        } for _ in range(num_class)
    ]
    for idx, img in enumerate(dataset.data):  # img.shape=(32,32,3)，即HWC
        cls = dataset.targets[idx]
        img_name = f"{idx}.jpg"
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(str(img_dir / img_name), img)
        if len(anns[cls]['sup']) < num_label_per_class:
            anns[cls]['sup'].append(f"{img_name} {cls}")
        else:
            anns[cls]['unsup'].append(f"{img_name} {cls}")
        # img = img.transpose(2, 0, 1)
        # cv2.imshow("demo", img)
        # cv2.waitKey(0)

    for class_anns in anns:
        assert len(class_anns['sup']) == num_label_per_class
        assert len(class_anns['unsup']) == len(dataset.targets) // num_class - num_label_per_class

    for task_idx in range(num_task):
        task_id = task_idx + 1
        ann_file_name = f"sup{num_label_per_class}_#{task_id}.txt"
        ann_unlabel_file_name = f"sup{num_label_per_class}_#{task_id}_unlabeled.txt"
        with open(str(ann_dir / ann_file_name), "w") as f1:
            with open(str(ann_dir / ann_unlabel_file_name), "w") as f2:
                for cls in range(task_idx * num_class_per_task, (task_idx+1)*num_class_per_task):
                    f1.writelines([x+"\n" for x in anns[cls]['sup']])
                    f2.writelines([x+"\n" for x in anns[cls]['unsup']])



if __name__ == "__main__":
    read_cifar10()
    exit(0)
