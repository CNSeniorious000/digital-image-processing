import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ==================================================================
# 3. 自行设计一个至少10层，准确率为93%以上的网络
# ==================================================================

# 判断GPU是否可用，配置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# 超参数设置
num_epochs = 15
num_classes = 10
batch_size = 128
learning_rate = 0.001

# 准备训练和测试数据集
train_dataset = torchvision.datasets.MNIST(root="../../data/", train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root="../../data/", train=False, transform=transforms.ToTensor())

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# ==================================================================
# 请完成这一部分，构建一个至少10层，准确率为93%以上的网络


# ==================================================================

model = ConvNet(num_classes).to(device)

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 训练网络
def train():
    model.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
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
    torch.save(model.state_dict(), "task3.ckpt")


if __name__ == "__main__":
    # summary(model, input_size=(1,28,28))

    train()
    test()
