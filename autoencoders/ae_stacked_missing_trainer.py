import numpy as np
import torch
from models.standard import StandardAutoencoder
from stacked_mnist import DataMode, StackedMNISTData
from torch.utils.data import DataLoader, TensorDataset


def ae_missing_trainer():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=9)

    train_data = gen.get_full_data_set()

    X_train, y_train = train_data

    X_train = (
        torch.from_numpy(X_train.astype(np.float32)).to(device).permute(0, 3, 1, 2)
    )
    y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = StandardAutoencoder(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = torch.nn.MSELoss()

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = loss_fn(reconstructed, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, loss: {total_loss}")

    torch.save(model.state_dict(), "trained_models/ae_stacked_missing.pth")


if __name__ == "__main__":
    ae_missing_trainer()
