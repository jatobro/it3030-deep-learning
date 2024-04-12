import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import DEVICE
from stacked_mnist import StackedMNISTData
from torch.utils.data import DataLoader


class StandardAE(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (28 * 28 -> 3)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )

        # decoder (3 -> 28 * 28)
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x


def train(model, loader, loss_fn, optimizer):
    losses = []

    model.train()
    for image, _ in loader:
        image = image.to(DEVICE)

        image = image.view(-1, 28 * 28)

        reconstructed = model(image)
        loss = loss_fn(reconstructed, image)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
    return losses


def get_image_reconstructed(model, loader):
    output = []

    model.eval()
    with torch.no_grad():
        for image, _ in loader:
            image = image.to(DEVICE)

            image = image.view(-1, 28 * 28)

            reconstructed = model(image)

            output.append((image, reconstructed))

    return output


def plot_image_reconstructed(images_reconstructed):
    for image, reconstructed in images_reconstructed:
        image = image.view(-1, 28, 28)
        reconstructed = reconstructed.view(-1, 28, 28)

        plt.figure()
        plt.imshow(image[1].cpu().numpy())
        plt.title("Original")
        plt.show()

        plt.figure()
        plt.imshow(reconstructed[1].cpu().numpy())
        plt.title("Reconstructed")
        plt.show()


if __name__ == "__main__":
    dataset = StackedMNISTData(root="data")
    loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=True)

    model = StandardAE().to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    losses = []
    for epoch in range(epochs):
        loss = train(model, loader, loss_fn, optimizer)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {sum(loss) / len(loss)}")

        losses.extend(loss)

    torch.save(model.state_dict(), "models/standard_ae.pth")

    plt.style.use("fivethirtyeight")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.plot(losses[-1000:])

    plt.show()
