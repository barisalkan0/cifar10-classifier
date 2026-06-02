import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ImprovedCNN


# test images need the same normalization values as training images
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)


# random crop and flip are not used during testing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])


def parse_args():
    # another checkpoint can be selected from the terminal if needed
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="improved_cifar10_model.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"model file was not found: {checkpoint}. run train.py first."
        )

    # train=False selects the test part of cifar-10
    test_data = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    # shuffling is not necessary during evaluation
    # using one worker is simpler on windows
    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load the weights that were saved after training
    model = ImprovedCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    correct = 0
    total = 0

    # gradients are not needed because weights are not updated here
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # max gives the class with the highest score
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
