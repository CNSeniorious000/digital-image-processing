import torch
from torch import Tensor, nn
from torch.optim.adam import Adam
from torchsummary import summary

from utils.mnist import cache_root, test_loader, train_loader
from utils.torch import device

# ==================================================================
# 1. 根据示例代码写出每一层的网络结构
# ==================================================================

# 超参数设置
num_epochs = 10
num_classes = 10
learning_rate = 0.001


# 构建卷积神经网络
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1_conv = nn.Sequential(nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.layer2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3_conv = nn.Sequential(nn.Conv2d(10, 32, kernel_size=5, stride=1, padding=0), nn.ReLU())
        self.layer4_fc = nn.Sequential(nn.Linear(10 * 10 * 32, 20), nn.ReLU())
        self.layer5_fc = nn.Sequential(nn.Linear(20, num_classes), nn.Softmax())

    def forward(self, x):
        out = self.layer1_conv(x)
        out = self.layer2_pool(out)
        out = self.layer3_conv(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer4_fc(out)
        out = self.layer5_fc(out)
        return out


model = ConvNet(num_classes).to(device)

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


# 训练网络
def train():
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")


# 测试模型
def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images: Tensor = images.to(device)
            labels: Tensor = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"测试集准确率为: {100 * correct / total} %")

    # 保存模型文件
    torch.save(model.state_dict(), cache_root / "task1.ckpt")


def main():
    summary(model, input_size=(1, 28, 28))
    train()
    test()
