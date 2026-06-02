import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ImprovedCNN


# mean and standard deviation values for cifar-10 channels
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)


# small image changes help the model learn better
# crop moves the image a little and flip mirrors it sometimes
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])


def parse_args():
    # these arguments are useful for both full training and short tests
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--checkpoint", default="improved_cifar10_model.pth")
    return parser.parse_args()


def main():
    args = parse_args()

    # torchvision downloads the dataset if it is not in the data folder
    train_data = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    # using one worker is simpler on windows
    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # move the model to gpu when cuda is available
    model = ImprovedCNN().to(device)

    # cross entropy loss is used for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # sgd updates the weights after backpropagation
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    # the learning rate slowly gets smaller during training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    for epoch in range(args.epochs):
        # training mode enables batch normalization updates
        model.train()
        total_loss = 0.0
        processed_batches = 0

        for batch_index, (images, labels) in enumerate(train_loader):
            # max-batches is only needed when a short test is wanted
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            # images and labels should be on the same device as the model
            images = images.to(device)
            labels = labels.to(device)

            # clear old gradients before calculating new ones
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # calculate gradients and update model weights
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            processed_batches += 1

        if processed_batches == 0:
            raise ValueError("--max-batches must be at least 1")

        # update learning rate once after each epoch
        scheduler.step()
        avg_loss = total_loss / processed_batches
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch + 1}/{args.epochs} | loss: {avg_loss:.4f} | lr: {current_lr:.6f}")

    torch.save(model.state_dict(), args.checkpoint)
    print(f"model saved: {args.checkpoint}")


if __name__ == "__main__":
    main()
