# LeNet-5 on FashionMNIST

## Overview
Implemented LeNet-5 using PyTorch and trained on the FashionMNIST dataset.

## Model
- Architecture: LeNet-5
- Input size: 28x28 grayscale images
- Activation: Sigmoid

## Training
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 20
- Batch size: 128

## Results
- Best validation accuracy: ~86%
- Training and validation curves shown below

## Files
- `models/lenet_5.py`: model definition
- `train/train_lenet_5.py`: training pipeline
- `results/`: saved model and curves

## Output
- `lenet_fashionmnist_best.pth`
- `lenet_fashionmnist_curves.png`