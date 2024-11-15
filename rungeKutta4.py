import torch
import torch.nn as nn
import torch.nn.functional as F

class RK4ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RK4ResidualBlock, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Step 0: Compute the shortcut output to match dimensions
        shortcut = self.shortcut(x)

        # Step 1: Compute k1
        k1 = F.relu(self.bn1(self.conv1(x)))

        # Step 2: Compute k2 using k1 and shortcut
        k2_input = shortcut + 0.5 * k1  # Match dimensions using shortcut
        k2 = F.relu(self.bn2(self.conv2(k2_input)))

        # Step 3: Compute k3 using k2 and shortcut
        k3_input = shortcut + 0.5 * k2  # Match dimensions using shortcut
        k3 = F.relu(self.bn3(self.conv3(k3_input)))

        # Step 4: Compute k4 using k3 and shortcut
        k4_input = shortcut + k3  # Match dimensions using shortcut
        k4 = F.relu(self.bn4(self.conv4(k4_input)))

        # RK4 combination: weighted sum of k1, k2, k3, k4
        rk4_output = (1/6) * (k1 + 2*k2 + 2*k3 + k4)

        # Add the shortcut connection to the RK4 output
        out = shortcut + rk4_output
        return out



class RK4ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(RK4ResNet, self).__init__()

        # Initial layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # Final fully connected layer
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(RK4ResidualBlock(in_channels, out_channels, stride))
        layers.append(RK4ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # Initial transformation
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out