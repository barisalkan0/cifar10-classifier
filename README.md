# CIFAR-10 Image Classifier

## Introduction

This project is an image classification model for the CIFAR-10 dataset. I built it with PyTorch to practice the main steps of a computer vision project, including data preparation, model design, training, and evaluation.

The model receives a small color image and predicts which class it belongs to. CIFAR-10 contains images from 10 different classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Project Goal

The first version of the project used a smaller CNN and reached 78.43% test accuracy. The goal was to improve this result by using a deeper network and a better training process.

The current model uses residual blocks, batch normalization, and data augmentation. After training, it reached 93.31% accuracy on the CIFAR-10 test set.

## Dataset

CIFAR-10 is a common dataset for image classification experiments.

- 50,000 training images
- 10,000 test images
- 10 classes
- 32 x 32 pixel RGB images

The dataset is downloaded automatically by torchvision when the training or evaluation script is run for the first time.

## Technologies Used

- Python
- PyTorch
- torchvision
- CUDA for GPU training

## Model Architecture

The model is a residual CNN written from scratch for CIFAR-10 images. It does not use a pretrained model.

The main structure is:

1. An initial convolution layer converts the RGB image into 64 feature maps.
2. Residual blocks learn image features at different levels.
3. Some blocks reduce the image size while increasing the number of channels.
4. Average pooling reduces the feature maps into a single feature vector.
5. A final linear layer produces scores for the 10 classes.

Residual blocks add a shortcut connection between the input and output of a block. This helps the model keep useful information while training a deeper network.

## Data Preparation

The training images use the following transformations:

- Random cropping with padding
- Random horizontal flipping
- Tensor conversion
- CIFAR-10 channel normalization

Random cropping and horizontal flipping create small variations of the training images. This helps the model learn general image features instead of memorizing the training set.

The test images only use tensor conversion and normalization. Random transformations are not used during evaluation.

## Training Process

The model is trained with:

- Cross entropy loss
- SGD optimizer
- Momentum
- Weight decay
- Cosine learning-rate scheduling
- 30 epochs by default
- Batch size of 128

The code automatically uses CUDA when a supported NVIDIA GPU is available. If CUDA is not available, it runs on the CPU.

## Setup

Clone the repository and open the project folder:

```powershell
git clone https://github.com/barisalkan0/cifar10-classifier.git
cd cifar10-classifier
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install PyTorch and torchvision:

```powershell
python -m pip install torch torchvision
```

For GPU training, install a CUDA-enabled PyTorch build by following the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

## Usage

Train the model:

```powershell
python train.py
```

The trained weights are saved as:

```text
improved_cifar10_model.pth
```

Evaluate the trained model:

```powershell
python evaluate.py
```

Run a short smoke test when checking the setup:

```powershell
python train.py --epochs 1 --max-batches 2 --checkpoint smoke_test_model.pth
python evaluate.py --checkpoint smoke_test_model.pth
```

## Result

The current model achieved:

```text
Test Accuracy: 93.31%
```

This result was measured on the 10,000 images in the CIFAR-10 test set.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch CIFAR-10 Dataset Documentation](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
- [PyTorch Image Transformations Documentation](https://docs.pytorch.org/vision/stable/transforms.html)
