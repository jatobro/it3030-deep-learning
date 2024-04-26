import matplotlib.pyplot as plt
import numpy as np
import torch
from models.standard import StandardAutoencoder
from stacked_mnist import DataMode, StackedMNISTData
from torch.utils.data import DataLoader, TensorDataset
from utils import to_one_hot


def ae_trainer():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=9)

    train_data = gen.get_full_data_set()
    test_data = gen.get_full_data_set(training=False)

    X_train, y_train = train_data
    X_test, y_test = test_data

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    X_train = (
        torch.from_numpy(X_train.astype(np.float32)).to(device).permute(0, 3, 1, 2)
    )
    y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_test = torch.from_numpy(X_test.astype(np.float32)).permute(0, 3, 1, 2)
    y_test = torch.from_numpy(y_test.astype(np.float32))

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32)

    model = StandardAutoencoder(in_channels=1).to(device)
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

    torch.save(model.state_dict(), "trained_models/ae.pth")

    model.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            reconstructed = model(images)
            if i == 0:
                break

    fig, axs = plt.subplots(2, 4, figsize=(10, 6), dpi=150)

    for i in range(2):
        for j in range(4):
            axs[i, j].imshow(reconstructed[j].squeeze().cpu().numpy(), cmap="gray")
            axs[i, j].axis("off")

    plt.show()


if __name__ == "__main__":
    ae_trainer()
