import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


# Define the neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)  # only two classes: face or not face

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


class BinaryCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        # Modify the label to be binary as needed
        if label == 1:
            label = 1
        else:
            label = 0

        return img, label


# Then use this new dataset instead of the original CIFAR10
trainset = BinaryCIFAR10(root="./data", train=True, download=True, transform=transform)
testset = BinaryCIFAR10(root="./data", train=False, download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

# Modify labels to have only two classes: face(1) and not-face(0)
# CIFAR-10's automobile class will be used as face for simplicity
for data in [trainset, testset]:
    for i in range(len(data)):
        if data[i][1] == 1:
            data[i] = (data[i][0], 1)
        else:
            data[i] = (data[i][0], 0)

# Initialize the model, loss function and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            logging.info(
                f"Epoch: {epoch + 1}, Batch: {i + 1}, Avg. Loss: {running_loss / 2000}"
            )
            running_loss = 0.0

logging.info("Finished Training")
