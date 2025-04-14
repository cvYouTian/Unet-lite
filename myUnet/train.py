import torch
import torchsummary
from Utils.set_seed import train_seed
from Utils.config import model_config
from network.model import Unet
from data.dataset import Images_Dataset_folder


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



if __name__ == '__main__':
    model = Unet(cfgs=model_cfg)
    model.to(device)

    torchsummary.summary(
        model, input_size=(3, para_cfg.input_size, para_cfg.input_size))
