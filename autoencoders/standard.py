import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import DEVICE
from stacked_mnist import StackedMNISTData
from torch.utils.data import DataLoader


class StandardAutoencoder(nn.Module):
    def __init__(self, in_channels=1, encoder_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, encoder_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(model, loader, loss_fn, optimizer):
    losses = []

    model.train()
    for image, _ in loader:
        image = image.to(DEVICE)

        reconstructed = model(image)
        loss = torch.sqrt(loss_fn(reconstructed, image))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
    return losses


def test(model, loader, loss_fn):
    losses = []

    model.eval()
    with torch.no_grad():
        for image, _ in loader:
            image = image.to(DEVICE)

            reconstructed = model(image)

            loss = torch.sqrt(loss_fn(reconstructed, image))
            losses.append(loss.item())

    return losses


def plot_most_anomalous(model, loader, loss_fn, k):
    most_anomalous = []

    model.eval()
    with torch.no_grad():
        for image, _ in loader:
            image = image.to(DEVICE)

            reconstructed = model(image)

            loss = torch.sqrt(loss_fn(reconstructed, image))

            if len(most_anomalous) < k:
                most_anomalous.append((loss, reconstructed))
            else:
                max_loss = max(most_anomalous, key=lambda x: x[0])[0]

                if loss < max_loss:
                    index = most_anomalous.index((max_loss, _))
                    most_anomalous[index] = (loss, reconstructed)

    for loss, reconstructed in most_anomalous:
        reconstructed = reconstructed.view(-1, 28, 28)

        plt.figure()
        plt.imshow(reconstructed[0].cpu().numpy())
        plt.title("Most anomalous")
        plt.show()


def get_image_reconstructed(model, loader):
    output = []

    model.eval()
    with torch.no_grad():
        for image, _ in loader:
            image = image.to(DEVICE)

            reconstructed = model(image)

            output.append((image, reconstructed))

    return output


if __name__ == "__main__":
    dataset = StackedMNISTData(root="data")
    loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=True)

    model = StandardAutoencoder().to(DEVICE)

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
