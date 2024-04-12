import torch
from config import DEVICE
from stacked_mnist import StackedMNISTData
from standard import (
    StandardAE,
    get_image_reconstructed,
    plot_image_reconstructed,
    train,
)
from torch.utils.data import DataLoader


def ae_basic():
    standard = StandardAE().to(DEVICE)

    try:
        standard.load_state_dict(torch.load("models/standard_ae.pth"))
        print("Pre-trained model found, using it...")
    except FileNotFoundError:
        print("No model found, training a new one...")

        train_dataset = StackedMNISTData(root="data", train=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(standard.parameters(), lr=1e-3)

        epochs = 10

        for epoch in range(epochs):
            loss = train(standard, train_loader, loss_fn, optimizer)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {sum(loss) / len(loss)}")

        torch.save(standard.state_dict(), "models/standard_ae.pth")

    test_dataset = StackedMNISTData(root="data", train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=True)

    plot_image_reconstructed(get_image_reconstructed(standard, test_loader))


if __name__ == "__main__":
    ae_basic()
