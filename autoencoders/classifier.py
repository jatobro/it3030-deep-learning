import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import DEVICE
from stacked_mnist import StackedMNISTData
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


def train(model, loader, loss_fn, optimizer):
    losses = []

    model.train()
    for image, label in loader:
        image, label = image.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()

        pred = model(image)

        loss = loss_fn(pred, label)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def test(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in loader:
            image, label = image.to(DEVICE), label.to(DEVICE)

            pred = model(image)

            _, predicted = torch.max(pred, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

    return correct / total


if __name__ == "__main__":
    train_dataset = StackedMNISTData(root="data")
    train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)

    test_dataset = StackedMNISTData(root="data", train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=True)

    model = Classifier().to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    losses = []
    for epoch in range(epochs):
        loss = train(model, train_loader, loss_fn, optimizer)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {sum(loss) / len(loss)}, Accuracy: {test(model, test_loader)}"
        )

        losses.extend(loss)

    torch.save(model.state_dict(), "models/verification_classifier.pth")

    plt.style.use("fivethirtyeight")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.plot(losses[-1000:])

    plt.show()
