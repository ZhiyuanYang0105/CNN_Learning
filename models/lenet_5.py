import torch
from torch import nn
from torchsummary import summary

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        # 第一层池化层
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 第二层池化层
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # flatten层
        self.flatten = nn.Flatten()
        # 全连接层
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    # 前向传播
    def forward(self, x):
        x = self.pool1(self.sigmoid(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool2(self.sigmoid(self.conv2(x)))   # 14x14 -> 5x5

        x = self.flatten(x)                        # 16*5*5 = 400

        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = LeNet5().to(device)

    from torchsummary import summary
    summary(model.to("cpu"), input_size=(1, 28, 28))
