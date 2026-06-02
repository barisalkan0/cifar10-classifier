import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # this convolution looks for basic image features
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # batch normalization keeps the values more stable
        # the second convolution uses the features from the first one
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # relu removes negative values
        self.relu = nn.ReLU(inplace=True)

        # the normal shortcut does not change the input
        self.shortcut = nn.Identity()

        # sometimes the image size or channel count changes
        # in that case the shortcut should have the same shape as the output
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # adding the shortcut is the main idea of a residual block
        out = out + shortcut
        out = self.relu(out)

        return out


class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # cifar-10 images have 3 color channels
        # the first part changes them into 64 feature maps
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # these blocks keep the image size as 32 by 32
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # stride=2 makes the image smaller and increases the channels
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )

        # average pooling gives one number for each feature map
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # there are 10 classes in cifar-10
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        # features pass through the layers in order
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        # flatten is needed before the linear layer
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x
