import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stacked_mnist import StackedMNISTData
from torch.utils.data import DataLoader

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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

        # print(f"Loss: {loss.item()}")
        losses.append(loss.item())
    return losses, image, reconstructed


if __name__ == "__main__":
    dataset = StackedMNISTData(root="data")
    loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=True)

    model = StandardAE().to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    losses = []
    images = []
    reconstructed_images = []
    for epoch in range(epochs):
        loss, image, reconstructed = train(model, loader, loss_fn, optimizer)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {sum(loss) / len(loss)}")

        losses.extend(loss)

        images.append(image)
        reconstructed_images.append(reconstructed)

    plt.style.use("fivethirtyeight")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.plot(losses[-100:])

    plt.show()

    with torch.no_grad():
        for image, reconstructed in zip(images, reconstructed_images):
            image = image.view(-1, 28, 28)
            reconstructed = reconstructed.view(-1, 28, 28)

            plt.figure()
            plt.imshow(image[0].cpu().numpy())
            plt.title("Original")
            plt.show()

            plt.figure()
            plt.imshow(reconstructed[0].cpu().numpy())
            plt.title("Reconstructed")
            plt.show()
