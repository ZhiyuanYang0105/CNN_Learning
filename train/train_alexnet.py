import os
import copy
import time

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


def train_val_data_process(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])

    full_data = FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_len = int(0.8 * len(full_data))
    val_len = len(full_data) - train_len

    train_data, val_data = Data.random_split(
        full_data,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = Data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs, lr=1e-4):
    device = get_device()
    print(f"Using device: {device}")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # ========== train ==========
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        for b_x, b_y in train_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            outputs = model(b_x)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(preds == b_y).item()
            train_num += b_x.size(0)

        train_loss_epoch = train_loss / train_num
        train_acc_epoch = train_corrects / train_num
        train_loss_all.append(train_loss_epoch)
        train_acc_all.append(train_acc_epoch)

        # ========== val ==========
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        with torch.no_grad():
            for b_x, b_y in val_dataloader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                outputs = model(b_x)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(preds == b_y).item()
                val_num += b_x.size(0)

        val_loss_epoch = val_loss / val_num
        val_acc_epoch = val_corrects / val_num
        val_loss_all.append(val_loss_epoch)
        val_acc_all.append(val_acc_epoch)

        if val_acc_epoch > best_acc:
            best_acc = val_acc_epoch
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Train Loss: {train_loss_epoch:.4f} | Train Acc: {train_acc_epoch:.4f}")
        print(f"Val   Loss: {val_loss_epoch:.4f} | Val   Acc: {val_acc_epoch:.4f}")
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    return model, train_loss_all, val_loss_all, train_acc_all, val_acc_all


if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)

    train_dataloader, val_dataloader = train_val_data_process(batch_size=64)

    model = AlexNet(num_classes=10)

    num_epochs = 10
    model, train_loss_all, val_loss_all, train_acc_all, val_acc_all = train_model_process(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs,
        lr=1e-4
    )

    model_path = "./results/alexnet_fashionmnist_best.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_all, label="Train Loss")
    plt.plot(epochs, val_loss_all, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_all, label="Train Acc")
    plt.plot(epochs, val_acc_all, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()

    fig_path = "./results/alexnet_fashionmnist_curves.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Curves saved to {fig_path}")

    plt.show()