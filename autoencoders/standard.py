import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import DEVICE
from stacked_mnist import StackedMNISTData
from torch.utils.data import DataLoader


class StandardAutoencoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.encoder_fc = nn.Linear(64 * 7 * 7, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        z = self.encoder(x)
        z = torch.flatten(z, 1)
        return self.encoder_fc(z)

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 7, 7)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


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
