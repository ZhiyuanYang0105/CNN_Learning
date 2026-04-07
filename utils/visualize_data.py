from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

def visualize():
    train_data = FashionMNIST(root="./data", 
                            train=True, 
                            download=True, 
                            transform=transforms.ToTensor())

    train_loader = data.DataLoader(train_data, 
                                batch_size=64, 
                                shuffle=True, 
                                num_workers=0)

    # 获取一个批次的数据
    images, labels = next(iter(train_loader))

    # 可视化一个批次的图像
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()

    # 定义标签名称
    label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  
         
    # 可视化64张图像
    plt.figure(figsize=(16, 8))
    for i in range(64):
        plt.subplot(8, 8, i + 1) # 创建8行8列的子图
        plt.imshow(images[i][0], cmap='gray')  # 显示图像
        plt.title(label_names[labels[i]])  # 显示标签名称
        plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.show()      

if __name__ == "__main__":
    visualize()
