from pathlib import Path
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import os
# from myUnet.Utils.config import model_config

# _, para_cfg= model_config()

class Images_Dataset_folder(Dataset):
    def __init__(self, image_dir, label_dir):

        self.image_dir = image_dir if isinstance(image_dir, Path) else Path(image_dir)
        self.label_dir = label_dir if isinstance(label_dir, Path) else Path(label_dir)

        # 因为label和image的文件名字是一样的，所以先进行sort将两个序列对齐
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))

        self.transI = torchvision.transforms.Compose([
            torchvision.transforms.Resize(2*[572]),
            torchvision.transforms.ToTensor(),
        ])
        self.transL = torchvision.transforms.Compose([
            torchvision.transforms.Resize(2*[388]),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.image_dir / self.images[idx]
        label_path = self.label_dir / self.labels[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)

        t_image = self.transI(image)
        t_label = self.transL(label)

        return t_image, t_label

if __name__ == '__main__':
    ...