import torch
from typing import Union
from pathlib import Path
import random
import numpy as np

def train_seed(seed: int):
    """
    设置的训练时所用的种子，确保实验的可复现性
    Args:
        seed:  种子

    Returns: None

    """
    # 设置cpu的种子
    torch.manual_seed(seed)
    # 设置单gpu的种子
    torch.cuda.manual_seed(seed)
    # 设置多gpu的种子
    torch.cuda.manual_seed_all(seed)
    # 设置numpy的种子
    np.random.seed(seed)
    # 设置python内置的random的随机种子
    random.seed(seed)
    # 确保使用cudnn时的卷积操作时确定的
    torch.backends.cudnn.deterministic = True
    # 关闭使用的benchmark自动优化算法，确保算法发的可复现性
    torch.backends.cudnn.benchmark = False

def data_seed(seed: int):
    pass

if __name__ == '__main__':
    pass