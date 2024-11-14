import torch
import torch.nn as nn
import torch.nn.functional as F

class RungeKuttaResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RungeKuttaResidualBlock, self).__init__()

        # First convolution layer (like the "half-step" in RK2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution layer (like the final step in RK2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Step 1: Compute the intermediate output (first RK step)
        k1 = F.relu(self.bn1(self.conv1(x)))

        # Step 2: Compute the second transformation based on k1 (second RK step)
        k2 = F.relu(self.bn2(self.conv2(k1)))

        # Use shortcut connection and add to second RK step output
        out = self.shortcut(x) + k2  # Final RK2 approximation
        return out

class RungeKuttaResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(RungeKuttaResNet, self).__init__()

        # Initial layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        layers.append(RungeKuttaResidualBlock(in_channels, out_channels, stride))
        layers.append(RungeKuttaResidualBlock(out_channels, out_channels, stride=1))
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

# Instantiate the model
model = RungeKuttaResNet(num_classes=10)
print(model)
