#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
为了调试方便，对原有标注文件生成包含图片更少的 mini 版
"""

from pathlib import Path
import json
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
from tqdm import tqdm

def gen_mini(ann_file, img_num):
    """

    :param ann_file:
    :param img_num: mini版标注文件需要多少图片
    :return:
    """
    ann_file = Path(ann_file)
    assert img_num > 0

    logger.info(f"正在打开并读取标注文件：{str(ann_file)}")
    with open(ann_file, 'r') as f1:
        source_data = json.load(f1)
        mini_data = {}
        img_cnt = 0
        for key in source_data:
            if key not in ['images']:
                mini_data[key] = source_data[key]
            else:
                mini_data['images'] = []
                for img in source_data['images']:
                    mini_data['images'].append(img)
                    img_cnt += 1
                    if img_cnt >= img_num:
                        break

        # 写入文件：
        mini_ann_file = ann_file.parent / f"{ann_file.stem}_mini.json"
        with open(mini_ann_file, "w") as f2:
            json.dump(mini_data, f2)


if __name__ == "__main__":
    gen_mini(
        "/mnt/f/coco/annotations/instances_val2017/instances_val2017_1.json",
        100
    )
    gen_mini(
        "/mnt/f/coco/annotations/instances_val2017/instances_val2017_2.json",
        100
    )
    gen_mini(
        "/mnt/f/coco/annotations/instances_val2017/instances_val2017_3.json",
        100
    )
    gen_mini(
        "/mnt/f/coco/annotations/instances_val2017/instances_val2017_4.json",
        100
    )
    gen_mini(
        "/mnt/f/coco/annotations/instances_val2017/instances_val2017_5.json",
        100
    )



    exit(0)
