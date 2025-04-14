import shutil
import time

import numpy as np
import torch
from torchsummary import torchsummary
import os
from torch.utils.data.sampler import SubsetRandomSampler
import random
from pathlib import Path
from Utils.set_seed import train_seed
from Utils.config import model_config
from network.model import Unet
from data.dataset import Images_Dataset_folder
from pytorch_run import y_pred
from pytorch_run_old import train_sampler, MAX_STEP, pred_tb

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# 设置训练的随机种子
train_seed(42)
# 使用固定算法，确保方法的可复现性
torch.use_deterministic_algorithms(True)

model_cfg, para_cfg= model_config()

model = Unet(cfgs=model_cfg)
model.to(device)

train_data = Images_Dataset_folder(para_cfg.image_path,
                                   para_cfg.label_path)
num_train = len(train_data)

indices = list(range(num_train))
split = int(np.floor(para_cfg.valid_size * num_train))

random_seed = random.randint(1, 100)

if para_cfg.shuffle:
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=para_cfg.batch_size,
                                           sampler=train_sampler,
                                           num_workers=para_cfg.num_workers,
                                           pin_memory=para_cfg.pin_memory)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=para_cfg.batch_size,
                                           sampler=valid_sampler,
                                           num_workers=para_cfg.num_workers,
                                           pin_memory_device=para_cfg.pin_memory)

initial_lr = para_cfg.initial_lr
# 定义Adam优化器
opt = torch.optim.Adam(model.parameters(), lr=initial_lr)

MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)

New_folder = Path(para_cfg.save_folder)

# 创建一个存放预测路径
read_pred = New_folder / Path("pred")
if read_pred.is_dir() and read_pred.exists():
    shutil.rmtree(read_pred)
else:
    read_pred.mkdir(parents=True, exist_ok=True)
# 创建一个存放权重的路径
read_model_path = New_folder / ...

if read_model_path.is_dir() and read_model_path.exists():
    shutil.rmtree(read_model_path)
else:
    read_model_path.mkdir(parents=True, exist_ok=True)

for i in range(para_cfg.epoch):
    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()
    scheduler.step(i)
    lr = scheduler.get_lr()

    model.train()
    k = 1

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # not imp
        input_images = (x, y, i, para_cfg.n_iter, k)

        opt.zero_grad()

        y_pred = model(x)
        lossT = ...



if __name__ == '__main__':
    model = Unet(cfgs=model_cfg)
    model.to(device)

    torchsummary.summary(
        model, input_size=(3, para_cfg.input_size, para_cfg.input_size))
