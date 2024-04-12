import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import DEVICE
from stacked_mnist import StackedMNISTData
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=14, latent_dim=200):
        super().__init__()

        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels * 2,
                out_channels=out_channels * 4,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels * 4,
                out_channels=out_channels * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * out_channels * 4, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, 28, 28)
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=14, latent_dim=200):
        super().__init__()

        self.out_channels = out_channels

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4 * out_channels * 7 * 7), nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(4 * out_channels, 4 * out_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                4 * out_channels,
                2 * out_channels,
                3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * out_channels, out_channels, 3, padding=1, stride=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 4 * self.out_channels, 7, 7)
        return self.conv(x)


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder.to(DEVICE)
        self.decoder = decoder.to(DEVICE)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x


def train(model, loader, loss_fn, optimizer):
    losses = []

    model.train()
    for image, _ in loader:
        image = image.to(DEVICE)

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

    model = Autoencoder(encoder=Encoder(), decoder=Decoder()).to(DEVICE)

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
