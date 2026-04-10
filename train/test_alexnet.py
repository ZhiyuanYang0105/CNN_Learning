import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import matplotlib.pyplot as plt

from models.alexnet import AlexNet


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_test_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])

    test_data = FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
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


def visualize_predictions(model, test_loader, num_images=16):
    device = get_device()
    label_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    images, labels = next(iter(test_loader))
    images_device = images.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images_device)
        _, preds = torch.max(outputs, 1)

    images = images.numpy()
    labels = labels.numpy()
    preds = preds.cpu().numpy()

    plt.figure(figsize=(12, 12))

    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i][0], cmap='gray', interpolation='nearest')

        true_label = label_names[labels[i]]
        pred_label = label_names[preds[i]]
        color = "green" if labels[i] == preds[i] else "red"

        plt.title(f"T: {true_label}\nP: {pred_label}", color=color, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_loader = load_test_data(batch_size=64)

    model = AlexNet(num_classes=10)
    state_dict = torch.load("./results/alexnet_fashionmnist_best.pth", map_location="cpu")
    model.load_state_dict(state_dict)

    test_model_process(model, test_loader)
    visualize_predictions(model, test_loader, num_images=16)