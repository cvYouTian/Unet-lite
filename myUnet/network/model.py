import torchsummary
import yaml
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union


class Conv(nn.Module):
    def __init__(self, in_ch, hi_ch, out_ch, k, s, p, c):
        super().__init__()

        self.down_conv = nn.Sequential(
            nn.Conv2d(in_ch, hi_ch, k, s, p, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hi_ch, out_ch, k, s, p, bias=True),
            nn.ReLU(inplace=True),
        )

        self.c = nn.MaxPool2d(kernel_size=2, stride=2) if c \
            else nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.c(x)

        return x


class Map(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class Unet(nn.Module):
    def __init__(self, cfgs):
        super(Unet, self).__init__()
        self.encoding = nn.ModuleList(
            Conv(ch[0], ch[1], ch[2], k, s, p, c) for ch, k, s, p, c in cfgs.encode
        )

        self.decoding = nn.ModuleList(
            Conv(ch[0], ch[1], ch[2], k, s, p, c) for ch, k, s, p, c in cfgs.decode
        )

        self.out = nn.ModuleList(
            Map(ch[0], ch[1], k, s, p) for ch, k, s, p in cfgs.out
        )

    def forward(self, x):
        # encode
        e = [x]
        e.extend(m(e[-1]) for m in self.encoding)
        e1 = F.interpolate(e[1], size=(392, 392))
        e2 = F.interpolate(e[2], size=(200, 200))
        e3 = F.interpolate(e[3], size=(104, 104))
        e4 = F.interpolate(e[4], size=(56, 56))

        # decode
        d = list(m for m in self.decoding)
        d1 = torch.cat((e4, d[0](e[-1])), 1)
        d2 = torch.cat((e3, d[1](d1)), 1)
        d3 = torch.cat((e2, d[2](d2)), 1)
        d4 = torch.cat((e1, d[3](d3)), 1)

        for m in self.out:
            d4 = m(d4)

        return d4


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config path
    yaml_path = Path("../configs/Unet.yaml")

    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    net = Unet(cfgs=config)
    net.to(device)

    torchsummary.summary(net, input_size=(3, 572, 572))