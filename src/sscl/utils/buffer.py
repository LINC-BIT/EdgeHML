
import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from copy import deepcopy
from mmdet.datasets.pipelines import Pad

def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)

        data_concatenate = torch.cat if type(dataset.train_loader.dataset.data) == torch.Tensor else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            refold_transform = lambda x: x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                refold_transform = lambda x: (x.cpu()*255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                refold_transform = lambda x: (x.cpu()*255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
            ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
            ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
                ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
                ])

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = [
            'examples',
            'labels', 'logits', 'task_labels',
            'img_metas', 'gt_bboxes', 'gt_labels',
            'sample_id'
        ]

    def to(self, device):
        # 因为涉及多GPU训练，且同时存在tensor数据和非tensor数据，
        # 所以统一存在cpu上
        raise NotImplementedError

        # self.device = device
        # for attr_str in self.attributes:
        #     if hasattr(self, attr_str) and attr_str != 'img_metas':
        #         setattr(self, attr_str, getattr(self, attr_str).to(device))
        # return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self,
                     examples: torch.Tensor,
                     labels: torch.Tensor=None,
                     logits: torch.Tensor=None,
                     task_labels: torch.Tensor=None,
                     img_metas=None,  # 目标检测用的
                     gt_bboxes=None,  # 目标检测用的
                     gt_labels=None,  # 目标检测用的
                     sample_id=None,

    ) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                # 目标检测任务中，图像的shape可能不同，
                # 此外，需要存储tensor数据和非tensor数据，
                # 所以更改buffer逻辑，统一存为list

                setattr(
                    self,
                    attr_str,
                    [None]*self.buffer_size
                )

                # typ = torch.int64 if attr_str.endswith('els') else torch.float32
                # setattr(
                #     self,
                #     attr_str,
                #     torch.zeros(
                #         (self.buffer_size, *attr.shape[1:]),
                #         dtype=typ, device=self.device
                #     )
                # )

    def add_data(self,
                 examples,
                 labels=None,  # 图像或box的类别
                 logits=None,  # 前馈计算后的类别概率
                 task_labels=None,  # 有些时候需要task_id
                 img_metas=None,  # 目标检测用的
                 gt_bboxes=None,  # 目标检测用的
                 gt_labels=None,  # 目标检测用的
                 sample_id=None,
    ):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(
                examples,
                labels, logits, task_labels,
                img_metas, gt_bboxes, gt_labels,
                sample_id
            )

        for i in range(examples.shape[0]):  # 将当前batch中的图像依次通过蓄水池采样判定是否放入buffer
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if img_metas is not None:
                    self.img_metas[index] = img_metas[i]  # 不需要to(device)
                if gt_bboxes is not None:
                    self.gt_bboxes[index] = gt_bboxes[i].to(self.device)
                if gt_labels is not None:
                    self.gt_labels[index] = gt_labels[i].to(self.device)
                if sample_id is not None:
                    self.sample_id[index] = sample_id[i]  # 不需要to(device)

    def get_data(self, size: int, transform: transforms=None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, len(self.examples)):
            size = min(self.num_seen_examples, len(self.examples))

        choice = np.random.choice(
            min(self.num_seen_examples, len(self.examples)),
            size=size,
            replace=False
        )
        # choice = np.random.choice(
        #     min(self.num_seen_examples, len(self.examples)),
        #     size=size,
        #     replace=True
        # )

        if transform is None:
            transform = lambda x: x

        # 目标检测任务中图片的size可能不同，所以需要pad：
        imgs = [transform(self.examples[cho].cpu()) for cho in choice]
        shape = list(imgs[0].shape)
        need_pad = False
        for i in range(1, len(imgs)):
            for dim in range(len(shape)):
                if imgs[i].shape[dim] != shape[dim]:
                    need_pad = True
                shape[dim] = max(shape[dim], imgs[i].shape[dim])
        if need_pad:
            new_imgs = [torch.zeros(shape, dtype=imgs[0].dtype) for _ in range(len(imgs))]
            for i in range(len(imgs)):
                new_imgs[i][0:imgs[i].shape[0], 0:imgs[i].shape[1], 0:imgs[i].shape[2]] = imgs[i]
            ret_tuple = (torch.stack(new_imgs).to(self.device), )
        else:
            ret_tuple = (torch.stack(imgs).to(self.device), )


        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                if attr_str in ['labels', 'logits', 'task_labels']:
                    ret_tuple += (torch.stack([attr[i] for i in choice]).to(self.device), )
                else:
                    ret_data = [attr[i] for i in choice]
                    if attr_str == 'img_metas':
                        for im in ret_data:
                            if isinstance(im, dict):
                                im['batch_input_shape'] = tuple(shape[1:])
                    ret_tuple += (ret_data, )

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

    def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        # raise NotImplementedError
        # assert isinstance(indexes, list) or isinstance(indexes, tuple)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                if hasattr(attr, 'to'):
                    attr = attr.to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (
            torch.stack([
                transform(ee.cpu()) for ee in self.examples if ee is not None
            ]).to(self.device)
            ,
        )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                if hasattr(attr, 'to'):
                    attr = attr.to(self.device)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        raise NotImplementedError
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
