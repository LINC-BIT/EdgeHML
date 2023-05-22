from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim


class BaseTask:
    """
    Semi-supervised Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass

    @staticmethod
    def get_minibatch_size():
        pass

from mmdet.datasets import DATASETS, ConcatDataset, build_dataset
from ssl_utils.models.softteacher.ssod.datasets import build_dataloader
from mmcls.datasets.base_dataset import BaseDataset, expanduser

class BaseSplitSemiTask(BaseDataset):

    @abstractmethod
    def __getitem__(self, index):
        pass

class ContinualLearningTasks:
    """
    Semi-supervised Continual learning Object Detection Task.
    """
    DS_NAME = None
    DS_SETTING = None
    DS_N_CLASSES_PER_TASK = None
    DS_N_TASKS = None

    def __init__(self, train_dataloaders, val_dataloaders=None) -> None:
        """
        将 mmdet 的 dataloaders 封装为持续学习的任务序列

        """
        assert isinstance(train_dataloaders, (list, tuple))
        assert isinstance(train_dataloaders[0], torch.utils.data.dataloader.DataLoader)

        self.train_data_loaders = train_dataloaders
        self.val_data_loaders = None
        if val_dataloaders is not None:
            self.set_val_dataloaders(val_dataloaders)


    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        pass

    def __getitem__(self, index):
        if index >= len(self.train_data_loaders):
            raise StopIteration(f"index >= {len(self.train_data_loaders)}")
        else:
            return self.train_data_loaders[index], self.val_data_loaders[index]

    def __len__(self):
        return len(self.train_data_loaders)

    def set_val_dataloaders(self, val_dataloaders):
        assert isinstance(val_dataloaders[0], torch.utils.data.dataloader.DataLoader)
        assert isinstance(val_dataloaders, (list, tuple))
        assert len(val_dataloaders) == len(self.train_data_loaders)
        self.val_data_loaders = val_dataloaders


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                         setting: BaseTask) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: BaseTask) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
