#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from mmcls.models import CLASSIFIERS
from datetime import datetime
import math
import multiprocessing
import pickle
# from ssl_utils.models.fixmatch.fixmatch_utils import consistency_loss

import torch
from torch.nn import functional as F
from mmcls.models import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier
from mmcls.models.losses import cross_entropy
from sscl.utils.buffer import Buffer
from sscl.models import BaseLearningMethod
import cv2
import numpy as np
import random
from ssl_utils.models.softteacher.ssod.utils import get_root_logger
from ssl_utils.train_utils import ce_loss
from sscl.utils.buffer import reservoir
import copy
from alipy.query_strategy.query_labels import QueryInstanceUncertainty, QueryInstanceQBC


class HML_v10:
    def __init__(
            self,
            num_classes,
            pool_sizes=(200, 2000),  # 内存池有标注数量、硬盘池容量
            device=torch.device("cpu")
    ):
        self.device = device

        self.mem_pool_sup = []
        self.mem_pool_sup_size = pool_sizes[0]
        self.mem_pool_sup_class_cnt = [0] * num_classes
        self.pool_indexes = {}
        self.hard_pool = []
        self.hard_pool_size = pool_sizes[1]
        self.mem_pool_unsup = []
        self.num_seen_unsup_samples = 0
        self.num_seen_sup_samples = 0

        self.shared_memory = None
        self.need_shared_memory = False

    def send2harddisk(self, obj):
        if self.need_shared_memory is True:
            if self.shared_memory is None:
                self.shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=1024 * 1024 * 1024)
            shared_object = pickle.dumps(obj)
            self.shared_memory.buf[:len(shared_object)] = shared_object
        else:
            torch.save(obj, "./edgehml_temp_data/dummy_img.pth")

    def push(self, img, attrs, cur_task_id: int):
        assert len(img.shape) == 3, "每次仅能添加1张图像，tensor不应包括channel维度"
        assert 'task_id' in attrs and 'img_index' in attrs

        cur_task_id = attrs['task_id']
        attrs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in attrs.items()
        }
        img = img.to(self.device)

        if 'is_pseudo' not in attrs or attrs['is_pseudo'] is False:
            this_class = int(attrs['label'].item())
            img_id = f"{cur_task_id}-{attrs['img_index']}"
            if img_id in self.pool_indexes:
                return
            attrs['img_id'] = img_id

            if len(self.mem_pool_sup) < self.mem_pool_sup_size:
                self.mem_pool_sup.append({
                    'img': img,
                    'attrs': attrs,
                })
                self.pool_indexes[img_id] = self.mem_pool_sup[-1]
                self.mem_pool_sup_class_cnt[this_class] += 1
            else:
                # raise NotImplementedError
                index = reservoir(self.num_seen_sup_samples, self.mem_pool_sup_size)
                if index >= 0:
                    old_img_id = self.mem_pool_sup[index]['attrs']['img_id']
                    old_class = int(self.mem_pool_sup[index]['attrs']['label'].item())
                    del self.pool_indexes[old_img_id]
                    self.mem_pool_sup_class_cnt[old_class] -= 1

                    self.mem_pool_sup[index] = {
                        'img': img,
                        'attrs': attrs,
                    }
                    self.pool_indexes[img_id] = self.mem_pool_sup[index]
                    self.mem_pool_sup_class_cnt[this_class] += 1

            self.num_seen_sup_samples += 1

        else:
            img_id = f"{cur_task_id}-{attrs['img_index']}-unsup"
            if img_id in self.pool_indexes:
                return
            attrs['img_id'] = img_id

            if len(self.hard_pool) < self.hard_pool_size:
                self.hard_pool.append({
                    'img': img,
                    'attrs': attrs,
                })
                self.pool_indexes[img_id] = self.hard_pool[-1]
                self.send2harddisk(self.hard_pool[-1])
            else:
                index = reservoir(self.num_seen_unsup_samples, self.hard_pool_size)
                if index >= 0:
                    self.hard_pool[index] = {
                        'img': img,
                        'attrs': attrs,
                    }
                    self.pool_indexes[img_id] = self.hard_pool[index]
                    self.send2harddisk(self.hard_pool[index])

            self.num_seen_unsup_samples += 1

    def get(self, cur_task_id: int, minibatch_size: int, is_pseudo: bool = False):

        # candidates = [i for i, sample in enumerate(self.mem_pool_sup + self.hard_pool)]

        if is_pseudo is False:
            if len(self.mem_pool_sup) == 0:
                return None

            indexes = np.random.choice(
                list(range(len(self.mem_pool_sup))),
                size=min(minibatch_size, len(self.mem_pool_sup)),
                replace=False
            )

            return [
                self.mem_pool_sup[i] for i in indexes
            ]
        else:
            if len(self.mem_pool_unsup) == 0:
                return None
            indexes = np.random.choice(
                list(range(len(self.mem_pool_unsup))),
                size=min(minibatch_size, len(self.mem_pool_unsup)),
                replace=False
            )

            return [
                self.mem_pool_unsup[i] for i in indexes
            ]


def consistency_loss(logits_s, logits_w, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    logits_w = logits_w.detach()
    if name == 'l2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean'), None, None, None

    elif name == 'l2_softmax':
        lw = torch.softmax(logits_w, dim=-1)
        ls = torch.softmax(logits_s, dim=-1)
        return F.mse_loss(ls, lw, reduction='mean'), None, None, None

    elif name == 'cos':
        lw = torch.softmax(logits_w, dim=-1)
        ls = torch.softmax(logits_s, dim=-1)
        return F.cosine_similarity(logits_s, logits_w).reciprocal().mean(), None, None, None

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        select = max_probs.ge(p_cutoff).long()

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long()

    else:
        raise NotImplementedError('Not Implemented consistency_loss')



@CLASSIFIERS.register_module(force=True)
class EdgehmlResnet18_v10(ImageClassifier, BaseLearningMethod):
    ALG_NAME = "EdgehmlResnet18_v10"  # 算法模型的名称
    ALG_COMPATIBILITY = ["class-il", "task-il"]

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(EdgehmlResnet18_v10, self).__init__(
                 backbone,
                 neck=neck,
                 head=head,
                 pretrained=pretrained,
                 train_cfg=train_cfg,
                 init_cfg=init_cfg
        )
        self.num_class = self.head.num_classes
        self.pool = HML_v10(
            self.num_class,
            train_cfg.pool_sizes,
            torch.device("cpu")
        )
        self.minibatch_size = train_cfg.minibatch_size
        self.minibatch_size_unsup = train_cfg.minibatch_size_unsup

        self.lambda_sup = train_cfg.lambda_sup

        self.lambda_sup_replay = train_cfg.lambda_sup_replay  # 有监督回放
        self.lambda_unsup_replay = 0  # 无监督回放
        self.raw_lambda_unsup_replay = train_cfg.lambda_unsup_replay
        self.need_sup_replay = train_cfg.need_sup_replay
        self.need_unsup_replay = train_cfg.need_unsup_replay

        self.raw_lambda_unsup = train_cfg.lambda_unsup
        self.lambda_unsup = 0
        self.T = 0.5
        self.p_cutoff = 0.95  # 置信度高于τ的prob（prob[i] = softmax(logits)[i]）

        self.logger = get_root_logger()
        self.num_class_per_task = None
        self.cur_task_class_index = None
        self.cur_task_id = -1
        self.cur_iter = 0
        self.unsup_iter_threshold = train_cfg.unsup_iter_threshold

        # 主动学习
        if hasattr(train_cfg, "active_learning") and len(str(train_cfg.active_learning)) > 0:
            self.active_learning = train_cfg.active_learning
        else:
            self.active_learning = None

        # num_param, num_trainable_param = count_parameters(self)
        # self.logger.info(f"DNN模型参数量、可训练参数量: {num_param} , {num_trainable_param}")

        self.threshold_length = train_cfg.threshold_length if hasattr(train_cfg, "threshold_length") else 0

        self.include_harddisk_read_time = train_cfg.include_harddisk_read_time


    def before_task(self, dataloaders, runner, *args, **kwargs):
        self.cur_task_class_index = dataloaders[0].dataset.class_index
        self.num_class_per_task = len(self.cur_task_class_index)

        if self.active_learning is not None:

            # 使用指定的主动学习算法重新采样标注样本
            self.logger.info(f"开始使用 {self.active_learning} 主动学习算法重新采样标注样本")
            ds = dataloaders[0].dataset
            candidates = {}
            for idx, img_info in enumerate(ds.unlabeled_data_infos):
                lab = img_info['gt_label'].item()
                if lab not in candidates:
                    candidates[lab] = []
                candidates[lab].append(
                    dict(
                        **(ds.sup_pipeline(
                            copy.deepcopy(img_info)
                        )),
                        global_id=idx
                    )
                )
            new_data_infos = []
            for gt_label, cands in candidates.items():
                features = torch.stack([x['img'] for x in candidates[gt_label]]).cuda()
                predicts = self._get_logits(features).detach().cpu().numpy()
                features = features.detach().cpu().numpy()
                # 基于不确定性的方法：
                if self.active_learning == "entropy" or self.active_learning == "least_confident" or self.active_learning == "margin":
                    # 参考：http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/huangsj/alipy/page_reference/api_classes/api_query_strategy.query_labels.QueryInstanceUncertainty.html
                    features = features.reshape(len(features), -1)
                    al = QueryInstanceUncertainty(
                        features,
                        [0] * len(features),
                        measure=self.active_learning
                    )
                    selected = al.select_by_prediction_mat(
                        [x['global_id'] for x in candidates[gt_label]],
                        predicts,
                        batch_size=ds.num_label_per_class
                    )
                # 基于委员会：
                elif self.active_learning == "qbc":
                    # 参考：http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/huangsj/alipy/page_reference/api_classes/api_query_strategy.query_labels.QueryInstanceUncertainty.html
                    al = QueryInstanceQBC(
                        features
                        [0] * len(features)
                    )
                    selected = al.select_by_prediction_mat(
                        [x['global_id'] for x in candidates[gt_label]],
                        predicts,
                        batch_size=ds.num_label_per_class
                    )
                else:
                    raise NotImplementedError
                for idx in selected:
                    new_data_infos.append(
                        copy.deepcopy(ds.unlabeled_data_infos[idx])
                    )
                self.logger.info(f"标注样本重新采样结果：类别 {gt_label}：{selected}")

            ds.data_infos = new_data_infos


    def after_task(self, *args, **kwargs):
        num_sup = len(self.pool.mem_pool_sup)
        num_unsup = len(self.pool.hard_pool)
        num_all = len(self.pool.mem_pool_sup) + len(self.pool.hard_pool)

        if num_all == 0:
            self.logger.info("当前 pool 为空")
            return

        if num_sup > 0:
            self.logger.info("当前内存池：")
            for i, c in enumerate(self.pool.mem_pool_sup_class_cnt):
                self.logger.info(f"类别 {i}:\t 样本 {c} 个,\t 占比 {round(c / num_sup * 100, 3)}%")

        if num_unsup > 0:
            tasks = []
            max_task_id = 0
            for samples in self.pool.hard_pool:
                tid = samples['attrs']['task_id']
                if tid not in tasks:
                    tasks.append(tid)
                    max_task_id = max(max_task_id, int(tid))
            num_per_task = [0] * (max_task_id+1)  # bugfix: 原为 [0] * len(tasks)
            for samples in self.pool.hard_pool:
                tid = samples['attrs']['task_id']
                num_per_task[int(tid)] += 1
            for tid, nu in enumerate(num_per_task):
                self.logger.info(f"无监督 - 任务 [{tid}]:\t 样本 {nu},\t 占比 {round(nu / num_unsup * 100, 3)}%")

        if self.lambda_unsup_replay > 0:
            self.logger.info(f"开始从硬盘池中采样样本并插入无监督内存池")

            # 对内存池中每个类别的loss进行统计：
            class_to_samples = {}
            for s in self.pool.mem_pool_sup:
                label = s['attrs']['label'].item()
                if label in class_to_samples:
                    class_to_samples[label].append(s)
                else:
                    class_to_samples[label] = [s]
            class_to_loss = {}
            for cls in class_to_samples:
                cur_class_imgs = torch.stack(
                    [s['img'] for s in class_to_samples[cls]]
                ).cuda()  # 此处原为 self.head.fc.weight.device，但是某些网络的head没有fc层，会报错
                cur_class_labels = torch.stack(
                    [s['attrs']['label'] for s in class_to_samples[cls]]
                ).cuda()
                cur_class_logits = self._get_logits(cur_class_imgs)
                cur_class_loss = self.head.loss(cur_class_logits, cur_class_labels)
                class_to_loss[cls] = cur_class_loss['loss'].detach().item()
            class_to_num = {
                k: 0 for k in class_to_loss
            }
            for s in self.pool.hard_pool:
                label = s['attrs']['label'].item()
                class_to_num[label] += 1
            class_to_prob = {
                k: len(self.pool.hard_pool) / max(v, 1) for k, v in class_to_num.items()
            }
            sum_loss = sum([v for k,v in class_to_loss.items()])
            for cls in class_to_prob:
                class_to_prob[cls] *= class_to_loss[cls] / sum_loss
            sum_prob = sum([v for k,v in class_to_prob.items()])
            final_class_to_prob = {
                k: v / sum_prob for k,v in class_to_prob.items()
            }
            p = [final_class_to_prob[s['attrs']['label'].item()] for s in self.pool.hard_pool]
            sum_p = sum(p)
            p = [x / sum_p for x in p]
            indexes = np.random.choice(
                list(range(len(self.pool.hard_pool))),
                size=min(
                    len(self.pool.hard_pool),
                    # self.pool.mem_pool_unsup_size,
                    self.pool.mem_pool_sup_size - len(self.pool.mem_pool_sup),
                ),
                replace=False,
                p=p
            )
            self.logger.info(f"已从硬盘池中采样出 {len(indexes)} 个标注样本：")
            final_num_per_class = {
                k: 0 for k in class_to_loss
            }
            for idx in indexes:
                self.pool.mem_pool_unsup.append(self.pool.hard_pool[idx])
                label = self.pool.hard_pool[idx]['attrs']['label'].item()
                final_num_per_class[label] += 1

                # read dummy sample object from disk:
                if self.include_harddisk_read_time is True:
                    try:
                        t = torch.load("./edgehml_temp_data/dummy_img.pth")
                    except Exception:
                        t = torch.jit.load("./edgehml_temp_data/dummy_img.pth")

            if self.include_harddisk_read_time is True and len(indexes > 0):
                self.logger.info(f"Dummy sample object: {t}")

            for k, v in final_num_per_class.items():
                self.logger.info(f"类别 [{k}]: {v} 个")

            self.logger.info(f"当前无监督内存池中样本：{len(self.pool.mem_pool_sup)}")

    def _get_logits(self, img):
        x = self.extract_feat(img)
        x = self.head.pre_logits(x)
        try:
            cls_score = self.head.fc(x)
        except Exception:
            # ViT的head没有fc层，需要使用如下方法：
            cls_score = self.head.layers.head(x)
        return cls_score

    def _sup_replay_loss(self, img, gt_label, cur_task_id, **kwargs):
        losses = {}
        # 数据回放loss：
        samples = None
        if self.lambda_sup_replay > 0:
            samples = self.pool.get(
                cur_task_id,
                self.minibatch_size,
            )
            if samples is not None and len(samples) > 0:
                buf_inputs = torch.stack([s['img'] for s in samples]).cuda()
                buf_labels = torch.stack([s['attrs']['label'] for s in samples]).cuda()

                buf_outputs = self._get_logits(buf_inputs)

                replay_loss = self.head.loss(buf_outputs, buf_labels)
                losses['replay_sup_loss'] = replay_loss['loss'] * self.lambda_sup_replay

        if self.lambda_sup_replay > 0 or self.need_sup_replay:
            for i, inp in enumerate(img):
                self.pool.push(inp, {
                    'label': gt_label[i],
                    'task_id': cur_task_id,
                    'img_index': kwargs['index'][i].item(),
                }, cur_task_id)

        return losses, samples

    def _unsup_contra_loss(self, cur_task_id, **kwargs):
        losses = {}
        # 无监督loss：
        weak_data = self.extract_unsup_cls_data(kwargs['unsup']['weak'])
        strong_data = self.extract_unsup_cls_data(kwargs['unsup']['strong'])
        logits_weak = self._get_logits(weak_data['img'])
        logits_strong = self._get_logits(strong_data['img'])

        if self.lambda_unsup > 0:
            unsup_loss, mask, select, pseudo_lb = consistency_loss(
                logits_strong,  # 无标注数据、强增强的logits
                logits_weak,  # 无标注数据，弱增强的logits
                'ce', self.T, self.p_cutoff,
                use_hard_labels=True)
            losses['unsup_loss'] = self.lambda_unsup * unsup_loss

        if self.lambda_unsup_replay > 0 or self.need_unsup_replay:
            for i, lw in enumerate(logits_weak):
                prob = torch.softmax(lw, dim=0)
                max_prob = torch.max(prob).item()
                if max_prob > self.p_cutoff:
                    max_prob_label = torch.argmax(prob)
                    if max_prob_label.item() in self.cur_task_class_index:
                        self.pool.push(weak_data['img'][i], {
                            'label': max_prob_label,
                            'is_pseudo': True,
                            'strong_data': strong_data['img'][i],
                            'task_id': cur_task_id,
                            'img_index': weak_data['index'][i],
                        }, cur_task_id)

        return losses

    def _unsup_replay_loss(self, cur_task_id, **kwargs):
        losses = {}
        # 数据回放loss：
        samples = None
        if self.lambda_sup_replay > 0:
            samples = self.pool.get(
                cur_task_id,
                self.minibatch_size,
                is_pseudo=True
            )
            if samples is not None and len(samples) > 0:
                buf_inputs = torch.stack([s['img'] for s in samples]).cuda()
                buf_labels = torch.stack([s['attrs']['label'] for s in samples]).cuda()

                buf_outputs = self._get_logits(buf_inputs)

                replay_loss = self.head.loss(buf_outputs, buf_labels)
                losses['replay_unsup_loss'] = replay_loss['loss'] * self.lambda_unsup_replay

        return losses, samples

    def _sup_ce_loss(self, img, gt_label, **kwargs):
        """
        该方法来自 mmcls\models\classifiers\image.py 中 ImageClassifier，
        因为需要额外返回 logits，所以重写该方法
        :param img:
        :param gt_label:
        :param kwargs:
        :return:
        """

        """
        为了提取 logits，需要分解 loss = self.head.forward_train(x, gt_label)，
        参考 mmcls\models\heads\linear_head.py 中的 LinearClsHead
        """

        assert len(img.shape) == 4
        losses = {}
        loss_arr = []
        cls_score = self._get_logits(img)
        sup_loss = ce_loss(cls_score, gt_label, reduction='mean')
        losses['sup_loss'] = sup_loss * self.lambda_sup

        return losses, cls_score, loss_arr

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

        task_id = int(kwargs['task_id'][0].item())
        losses = {}

        if task_id != self.cur_task_id:
            self.cur_task_id = task_id
            self.cur_iter = 0
        else:
            self.cur_iter += 1

        if self.cur_iter < self.unsup_iter_threshold:
            self.lambda_unsup = 0
        elif self.threshold_length > 0 and self.unsup_iter_threshold <= self.cur_iter < self.unsup_iter_threshold+self.threshold_length:
            self.lambda_unsup = -0.5 * math.cos(math.pi * (self.cur_iter - self.unsup_iter_threshold) / self.threshold_length) + 0.5
            assert 0 <= self.lambda_unsup <= 1
        else:
            self.lambda_unsup = self.raw_lambda_unsup

        self.lambda_unsup_replay = self.raw_lambda_unsup_replay

        # 正常有监督的loss：
        sup_losses, sup_logits, sup_loss_arr = self._sup_ce_loss(img, gt_label, **kwargs)
        losses.update(sup_losses)


        # 有监督回放loss：
        sup_replay_losses, replay_samples = self._sup_replay_loss(img, gt_label, task_id, **kwargs)
        losses.update(sup_replay_losses)

        # 无监督loss：
        unsup_losses = self._unsup_contra_loss(task_id, **kwargs)
        losses.update(unsup_losses)

        unsup_replay_losses, unsup_replay_samples = self._unsup_replay_loss(task_id, **kwargs)
        losses.update(unsup_replay_losses)

        return losses


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable_num


@CLASSIFIERS.register_module(force=True)
class EdgehmlMobilenetv2(EdgehmlResnet18_v10):
    ALG_NAME = "EdgehmlMobilenetv2"
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
            optimizer_config = copy.deepcopy(runner.optimizer.defaults)
            optimizer_config['params'] = self.net.parameters()
            del runner.optimizer
            runner.optimizer = obj_cls(**optimizer_config)
            self.logger.info(f"已更换 optimizer：对新的 net 使用 defaults config 构造了 optimizer")

            # 测定DNN规模：
            num_param, num_trainable_param = count_parameters(self.net)
            self.logger.info(f"DNN模型参数量、可训练参数量: {num_param} , {num_trainable_param}")

        super().before_task(dataloaders, runner, *args, **kwargs)

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

    def _get_logits(self, img):
        x = self.net(img)
        return x


@CLASSIFIERS.register_module(force=True)
class EdgehmlRAN(EdgehmlMobilenetv2):
    ALG_NAME = "EdgehmlRAN"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.residual_attention_network import ResidualAttentionModel_92_32input_update as RAN
        return RAN(num_classes)


@CLASSIFIERS.register_module(force=True)
class EdgehmlCBAM(EdgehmlMobilenetv2):
    ALG_NAME = "EdgehmlCBAM"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.cbam import cbamnet
        return cbamnet(network_type='CIFAR100', num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class EdgehmlResnext29(EdgehmlMobilenetv2):
    ALG_NAME = "EdgehmlResnext29"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.resnext import resnext29_2x64d
        return resnext29_2x64d(num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class EdgehmlInceptionv3(EdgehmlMobilenetv2):
    ALG_NAME = "EdgehmlInceptionv3"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.inception import inceptionv3
        return inceptionv3(num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class EdgehmlSeresnet18(EdgehmlMobilenetv2):
    ALG_NAME = "EdgehmlSeresnet18"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.senet import senet18
        return senet18(num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class EdgehmlVGG16(EdgehmlMobilenetv2):
    ALG_NAME = "EdgehmlVGG16"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.vgg import vgg16
        return vgg16(num_classes=num_classes)



