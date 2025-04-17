import shutil
import time
import numpy as np
import torch
from torchsummary import torchsummary
from torch.utils.data.sampler import SubsetRandomSampler
import random
from pathlib import Path
from Utils.set_seed import train_seed
from Utils.config import model_config
from network.model import Unet
from data.dataset import Images_Dataset_folder
from network import loss
from tqdm import tqdm


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


valid_loss_min = np.Inf
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)

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
                                           pin_memory=para_cfg.pin_memory)

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
read_model_path = New_folder / Path("Unet_"+ str(para_cfg.epoch) + '_' + str(para_cfg.batch_size))

if read_model_path.is_dir() and read_model_path.exists():
    shutil.rmtree(read_model_path)
else:
    read_model_path.mkdir(parents=True, exist_ok=True)


for i in range(para_cfg.epoch):
    print(i)
    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()


    model.train()

    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        opt.zero_grad()

        y_pred = model(x)
        # dice loss
        lossT = loss.calc_loss(y_pred, y)
        # 将平均损失变成此批次的总损失
        train_loss += lossT.item() * x.size(0)

        lossT.backward()
        opt.step()

    scheduler.step()
    # aqirue last loss
    lr = scheduler.get_last_lr()


    model.eval()
    torch.no_grad()

    for x, y in tqdm(valid_loader):
        x, y = x.to(device), y.to(device)

        y_pred = model(x)
        lossL = loss.calc_loss(y_pred, y)

        valid_loss += lossL.item() * x.size(0)

    train_loss /= len(train_idx)
    valid_loss /= len(valid_idx)

    if (i + 1) % para_cfg.print_every == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            i + 1, para_cfg.epoch, train_loss, valid_loss))

    # 如果验证集的损失比最小的损失小才会保存
    if valid_loss <= valid_loss_min:
        print("Validation Loss decreased ({:.6f} --> {:.6f}). Saving model...".format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), str(read_model_path / Path("123.pth")))

        # if round(valid_loss, 4) == round(valid_loss_min, 4):
        #     print(i_valid)
        #     i_valid += 1
        valid_loss_min = valid_loss



if __name__ == '__main__':
    model = Unet(cfgs=model_cfg)
    model.to(device)

    torchsummary.summary(
        model, input_size=(3, para_cfg.input_size, para_cfg.input_size))
