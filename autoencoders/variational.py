import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import DEVICE
from stacked_mnist import StackedMNISTData
from torch.utils.data import DataLoader


class VariationalAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder.to(DEVICE)
        self.decoder = decoder.to(DEVICE)

    def reparameterize(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)

        return self.decoder(z), mean, logvar


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )

        self.mean = nn.Linear(latent_dim, 2)
        self.logvar = nn.Linear(latent_dim, 2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        z = self.encoder(x)
        mean = self.mean(z)
        logvar = self.logvar(z)

        return mean, logvar


class VariationalDecoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x).view(-1, 1, 28, 28)


def vae_loss_fn(target, pred, mean, logvar):
    reproduction_loss = nn.functional.binary_cross_entropy(
        pred, target, reduction="sum"
    )
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return reproduction_loss + kld


def train(model, loader, optimizer):
    model.train()

    losses = []
    for X, _ in loader:
        X = X.to(DEVICE)

        optimizer.zero_grad()

        pred, mean, logvar = model(X)

        loss = vae_loss_fn(X, pred, mean, logvar)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    avg_loss = sum(losses) / len(losses)
    return avg_loss


def test(model, loader):
    model.eval()

    losses = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(DEVICE)

            pred, mean, logvar = model(X)

            loss = vae_loss_fn(X, pred, mean, logvar)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return avg_loss


def get_variational_image_reconstructed(model, loader):
    output = []

    model.eval()
    with torch.no_grad():
        for image, _ in loader:
            image = image.to(DEVICE)

            reconstructed, _, _ = model(image)

            output.append((image, reconstructed))

    return output


if __name__ == "__main__":
    train_dataset = StackedMNISTData(root="data")
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    model = VariationalAE().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50

    avg_losses = []
    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer)
        print(f"Epoch {epoch+1}, Avg Loss: {loss}")

        avg_losses.append(loss)

    torch.save(model.state_dict(), "variational_ae.pth")

    plt.plot(avg_losses)
    plt.show()
