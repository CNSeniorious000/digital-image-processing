import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary

# ==================================================================
# 1. 根据示例代码写出每一层的网络结构    
# ==================================================================

# 判断GPU是否可用，配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 100
batch_size = 128
sample_dir = 'samples'

# 创建文件夹
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# 图像预处理
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])

# 创建数据集
mnist = torchvision.datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    download=True)

# 创建数据加载器
data_loader = torch.utils.data.DataLoader(
    dataset=mnist,
    batch_size=batch_size, 
    shuffle=True)

# 构建生成器
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# 构建判别器
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# 将模型放入到对应的设备中
G = G.to(device)
D = D.to(device)

# 定义损失函数与优化器
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

# 开始训练
def train():
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(data_loader):
            imgs = imgs.reshape(-1, image_size).to(device)

            # 真实图像标签为 1，假图像标签为 0
            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)

            # 训练判别器
            d_optimizer.zero_grad()
            outputs = D(imgs)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(imgs.size(0), latent_size).to(device)
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
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], '
                    f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
        
        # 保存真实图像
        if (epoch+1) == 1:
            imgs = imgs.reshape(imgs.size(0), 1, 28, 28)
            save_image(imgs.clamp(0, 1), os.path.join(sample_dir, 'real_images.png'))
        
        # 保存生成的假图像
        if (epoch+1) % 10 == 0:
            fake_imgs = fake_imgs.reshape(fake_imgs.size(0), 1, 28, 28)
            save_image(fake_imgs.clamp(0, 1), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

    # 保存模型
    torch.save(G.state_dict(), 'G_1.ckpt')
    torch.save(D.state_dict(), 'D_1.ckpt')

if __name__ == "__main__":
    # 查看模型结构与参数
    summary(G, input_size=(1, 64))
    summary(D, input_size=(1, 784))

    train()