from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from utils.mnist import cache_root
from utils.torch import device

# ==================================================================
# 2. 根据给出的生成器结构，构建带有卷积层的判别器
# ==================================================================


# 超参数设置
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 100
batch_size = 128
sample_dir = "samples-2"

# 创建文件夹
if not Path(sample_dir).exists():
    Path(sample_dir).mkdir()

# 图像预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

# 创建数据集
mnist = torchvision.datasets.MNIST(root=cache_root, train=True, transform=transform, download=True)

# 创建数据加载器
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1_deconv = nn.Sequential(nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2), nn.BatchNorm2d(16), nn.ReLU())
        self.layer2_deconv = nn.Sequential(nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2), nn.BatchNorm2d(32), nn.ReLU())
        self.layer3_deconv = nn.Sequential(nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer4_deconv = nn.Sequential(nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2), nn.Tanh())

    def forward(self, x):
        out = self.layer1_deconv(x)
        out = self.layer2_deconv(out)
        out = self.layer3_deconv(out)
        out = self.layer4_deconv(out)
        return out


G = Generator()
# ==================================================================


# ==================================================================
# 请完成这一部分，构建判别器 D
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1_conv = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.layer2_conv = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.layer3_conv = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2))
        self.layer4_conv = nn.Sequential(nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=0), nn.Sigmoid())

    def forward(self, x):
        out = self.layer1_conv(x)
        out = self.layer2_conv(out)
        out = self.layer3_conv(out)
        out = self.layer4_conv(out)

        return out


D = Discriminator()

# ==================================================================

# 将模型放入到对应的设备中
D = D.to(device)
G = G.to(device)

# 定义损失函数与优化器
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


# 开始训练
def train():
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(data_loader):
            imgs = imgs.to(device)

            # 创建标签
            real_labels = torch.ones(imgs.shape[0], 1, 1, 1).to(device)
            fake_labels = torch.zeros(imgs.shape[0], 1, 1, 1).to(device)

            # 训练判别器
            d_optimizer.zero_grad()
            outputs = D(imgs)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(imgs.shape[0], latent_size, 1, 1).to(device)
            fake_imgs = G(z)
            outputs = D(fake_imgs.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            outputs = D(fake_imgs)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], "
                    f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
                )

        # 保存真实图像
        if (epoch + 1) == 1:
            imgs = imgs.reshape(imgs.size(0), 1, 28, 28)
            save_image(imgs.clamp(0, 1), Path(sample_dir, "real_images.png"))

        # 保存生成的假图像
        if (epoch + 1) % 10 == 0:
            fake_imgs = fake_imgs.reshape(fake_imgs.size(0), 1, 28, 28)
            save_image(fake_imgs.clamp(0, 1), Path(sample_dir, f"fake_images-{epoch + 1}.png"))

    # 保存模型
    torch.save(G.state_dict(), "G.ckpt")
    torch.save(D.state_dict(), "D.ckpt")


if __name__ == "__main__":
    # summary(G, input_size = (64, 1, 1))
    # summary(D, input_size = (1, 28, 28))

    train()
