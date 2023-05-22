
import os
import inspect
import importlib
from .utils.gcl_dataset import GCLDataset
from .utils.base_task import BaseTask
from argparse import Namespace

from .image_detection.split_coco import *

def get_all_models():
    return [model.split('.')[0] for model in os.listdir(
        os.path.dirname(__file__)
    ) if not model.find('__') > -1 and 'py' in model]

NAMES = {}
for model in get_all_models():
    mod = importlib.import_module("sscl.datasets." + model)

    dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'BaseTask' in str(inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c
    
    gcl_dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'GCLDataset' in str(inspect.getmro(getattr(mod, x))[1:])]
    for d in gcl_dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c

def get_dataset(args: Namespace) -> BaseTask:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
