# CIFAR-10 Image Classifier

A simple CNN-based image classifier built with PyTorch, trained on the CIFAR-10 dataset.

## About
- Built a Convolutional Neural Network (CNN) from scratch using PyTorch
- Trained on 60,000 images across 10 categories (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- Achieved **73.84% accuracy** on 10,000 test images

## Requirements
```bash
pip install torch torchvision matplotlib
```

## Usage
```bash
# Train the model
python train.py

# Evaluate accuracy
python evaluate.py
```

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 73.84% |
| Epochs | 10 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |