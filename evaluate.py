import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ImprovedCNN

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=128, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load('cifar10_model.pth', map_location=device))
model.eval()

correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / len(test_data)
print(f"Test Accuracy: {accuracy:.2f}%")