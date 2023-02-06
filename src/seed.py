import os 
import random
import numpy as np 
import torch

DEFAULT_RANDOM_SEED = 2021

def seedBasic(seed=42):
    """Sets seed for basic libraries

    Args:
        seed (int): Seed to set

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def seedTorch(seed=42):
    """Sets seed Pytorch library

    Args:
        seed (int): Seed to set

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
def seedEverything(seed=42):
    """Sets seed for everything

    Args:
        seed (int): Seed to set

    Returns:
        None
    """
    seedBasic(seed)
    seedTorch(seed)