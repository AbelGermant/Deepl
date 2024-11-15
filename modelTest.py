from euler import ResNet
from rungeKutta import RungeKuttaResNet
from rungeKutta4 import RK4ResNet

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 5
num_classes = 10  # MNIST has 10 digit classes

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # ResNet expects 32x32 input, so resize
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define function to train and test models
def train_and_evaluate(model, model_name):
    print(f"Training {model_name}...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"{model_name} Accuracy on the MNIST test set: {accuracy:.2f}%\n")

# Instantiate and test both models
resnet_model = ResNet(num_classes=num_classes)
rk_resnet_model = RungeKuttaResNet(num_classes=num_classes)
rk4_resnet_model = RK4ResNet(num_classes=num_classes)

#train_and_evaluate(resnet_model, "ResNet")
#train_and_evaluate(rk_resnet_model, "Runge-Kutta ResNet")
train_and_evaluate(rk4_resnet_model, "RK4 ResNet")