import torch

print("Torch version:", torch.__version__)

# 判断设备
if torch.backends.mps.is_available():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("Device:", device)

# 测试张量
x = torch.rand(3, 3).to(device)
print(x)