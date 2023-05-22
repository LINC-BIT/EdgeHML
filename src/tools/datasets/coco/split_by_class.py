#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
对于目标检测的 annotations（例如 instances_val2017.json），
根据类别划分为多个标注文件（每个文件作为一个 task 用于持续学习），
注意，同一张图像可能出现在多个 task 中，但在不同 task 中 bbox 的类别不同。
"""

from pathlib import Path
import json
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
from tqdm import tqdm

def split(tasks_classes, ann_file, target_dir=None):
    """

    :param tasks_classes:
    :param ann_file:
    :param target_dir: 为 None 时，自动在 ann_file 同级目录中
                        创建子目录，存放各个 task 的标注文件
    :return:
    """
    ann_file = Path(ann_file)
    if target_dir is None:
        target_dir = ann_file.parent / ann_file.stem
    if not target_dir.exists() or not target_dir.is_dir():
        target_dir.mkdir(parents=True)

    logger.info(f"正在打开并读取标注文件：{str(ann_file)}")
    with open(ann_file, 'r') as f1:
        source_data = json.load(f1)
        categories = source_data['categories']

        for task_id, classes in enumerate(tasks_classes):
            logger.info(f"正在处理第 [{task_id+1}/{len(tasks_classes)}] 个 task 的数据...")

            # 获取当前task所有类别的 id：
            classes_ids = []
            for cat in categories:
                if cat['name'] in classes:
                    classes_ids.append(cat['id'])

            # 提取当前task的数据：
            cur_task_data = {}
            for key in source_data:
                if key not in ['annotations']:
                    cur_task_data[key] = source_data[key]
                else:
                    # 对于 annotations 数据，
                    cur_task_data['annotations'] = []
                    logger.info(f"开始提取 bbox...")
                    bbox_cnt = 0
                    for bbox in tqdm(source_data['annotations']):
                        if bbox['category_id'] in classes_ids:
                            cur_task_data['annotations'].append(bbox)
                            bbox_cnt += 1
                    logger.info(f"当前 task_{task_id+1} 的 bbox 总计：{bbox_cnt} 个")

            # 写入文件：
            task_ann_file = target_dir / f"{ann_file.stem}_{task_id+1}.json"
            with open(task_ann_file, "w") as f2:
                json.dump(cur_task_data, f2)


if __name__ == "__main__":
    tasks_classes = [
        # task_1:
        ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench"],
        # task_2:
        ["backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","chair","couch","potted plant","bed","dining table","toilet"],
        # task_3:
        ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
        # task_4:
        ["banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake"],
        # task_5:
        ["tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"],
    ]
    split(
        tasks_classes,
        "/mnt/f/coco/annotations/semi_supervised/instances_train2017.1@1.json"
    )
    split(
        tasks_classes,
        "/mnt/f/coco/annotations/semi_supervised/instances_train2017.1@1-unlabeled.json"
    )
    split(
        tasks_classes,
        "/mnt/f/coco/annotations/instances_train2017.json"
    )



    exit(0)
