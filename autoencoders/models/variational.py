import torch
import torch.nn as nn


class VariationalAE(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.device = device

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def reparameterize(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
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
