

import random
import torch
import numpy as np

def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to save dataset,
    such as cifar-10-batches-py's parent dir.
    """
    # return '/data/wxw/datasets/'
    return '/mnt/e/SSCL_exp/DER/'


def base_log_path() -> str:
    """
    Returns the base bath where to log tensorboard.
    """
    # return './run_data/'
    return '/mnt/e/SSCL_exp/DER/'


def set_random_seed(seed: int, deterministic=True) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 补充：
    if deterministic:
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
