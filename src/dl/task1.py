import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ==================================================================
# 1. 根据示例代码写出每一层的网络结构
# ==================================================================

# 判断GPU是否可用，配置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 超参数设置
num_epochs = 10
num_classes = 10
batch_size = 128
learning_rate = 0.001

# 准备训练和测试数据集
train_dataset = torchvision.datasets.MNIST(root="../../data/", train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root="../../data/", train=False, transform=transforms.ToTensor())

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# 构建卷积神经网络
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


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
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"测试集准确率为: {100 * correct / total} %")

    # 保存模型文件
    torch.save(model.state_dict(), "task1.ckpt")


if __name__ == "__main__":
    # 查看网络结构和参数
    # summary(model, input_size=(1,28,28))

    train()
    test()
