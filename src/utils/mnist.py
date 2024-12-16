from pathlib import Path

import torchvision.transforms as transforms
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

cache_root = Path(__file__, "../../../cache")

train_dataset = MNIST(root=cache_root, train=True, transform=transforms.ToTensor(), download=True)

test_dataset = MNIST(root=cache_root, train=False, transform=transforms.ToTensor())


BATCH_SIZE = 128

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_loader: DataLoader[Image] = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2)
