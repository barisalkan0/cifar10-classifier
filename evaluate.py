import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64)

model = SimpleCNN()
model.load_state_dict(torch.load('cifar10_model.pth'))
model.eval()

correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / len(test_data)
print(f"Test Accuracy: {accuracy:.2f}%")