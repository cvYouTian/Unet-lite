from pathlib import Path
from typing import Union
import torch.nn.functional as F
from network.model import Unet
import torchvision
import matplotlib.pyplot as plt
import torch.cuda
from Utils.config import model_config
from PIL import Image


device = "gpu" if torch.cuda.is_available() else "cpu"

model_cfg, para_cfg = model_config()


# setting
input_size = 2*[para_cfg.input_size]
test_image = Path(para_cfg.test_image)
test_label = Path(para_cfg.test_label)

data_transform = torchvision.transforms.Compose([
             torchvision.transforms.Resize(input_size),
            torchvision.transforms.ToTensor(),
        ])

model = Unet(cfgs=model_cfg)
model.to(device)

model.load_state_dict(torch.load('xxx.pth'))
#
im_tb = Image.open(str(test_image))
im_label = Image.open(str(test_label))
s_tb = data_transform(im_tb)
s_label = data_transform(im_label)
s_label = s_label.detach().numpy()

pred_tb = model(s_tb.unsqueeze(0).to(device)).cpu()
pred_tb = F.sigmoid(pred_tb)
pred_tb = pred_tb.detach().numpy()

x1 = plt.imsave('./model/pred/img_iteration_test.png', pred_tb[0][0])
