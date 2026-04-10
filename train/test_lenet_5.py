import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from models.lenet_5 import LeNet5


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_test_data():
    test_data = FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    return test_loader


def test_model_process(model, test_loader):
    device = get_device()
    print(f"Using device: {device}")

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    test_corrects = 0
    test_num = 0

    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            outputs = model(b_x)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, b_y)

            test_loss += loss.item() * b_x.size(0)
            test_corrects += torch.sum(preds == b_y).item()
            test_num += b_x.size(0)

    test_loss_epoch = test_loss / test_num
    test_acc_epoch = test_corrects / test_num

    print(f"Test Loss: {test_loss_epoch:.4f}")
    print(f"Test Acc:  {test_acc_epoch:.4f}")

    return test_loss_epoch, test_acc_epoch


if __name__ == "__main__":
    test_loader = load_test_data()

    model = LeNet5()
    model.load_state_dict(torch.load("./results/lenet_fashionmnist_best.pth", map_location="cpu"))

    test_model_process(model, test_loader)