import matplotlib.pyplot as plt
import numpy as np
import torch
from models.standard import StandardAutoencoder
from stacked_mnist import DataMode, StackedMNISTData
from torch.utils.data import DataLoader, TensorDataset


def ae_anom():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = StandardAutoencoder(in_channels=3).to(device)
    model.load_state_dict(torch.load("trained_models/ae_stacked_missing.pth"))

    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=9)
    test_data = gen.get_full_data_set(training=False)

    X_test, y_test = test_data

    X_test = torch.from_numpy(X_test.astype(np.float32)).permute(0, 3, 1, 2)
    y_test = torch.from_numpy(y_test.astype(np.float32))

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    model.eval()

    criterion = torch.nn.BCELoss()

    anomalous_images = []
    max_losses = []
    labels = []

    with torch.no_grad():
        for x, y in test_loader:
            y = y.argmax(dim=0).item()
            x = x.to(device)
            pred = model(x)
            loss = criterion(pred, x)
            loss = loss.item()
            if len(max_losses) > 7:
                min_val = min(max_losses)
                min_loss_index = max_losses.index(min_val)
                if loss > min_val:
                    anomalous_images[min_loss_index] = pred.squeeze(0).cpu().numpy()
                    max_losses[min_loss_index] = loss
                    labels[min_loss_index] = y
            else:
                anomalous_images.append(pred.squeeze(0).cpu().numpy())
                max_losses.append(loss)
                labels.append(y)

    fig, axs = plt.subplots(2, 4, figsize=(10, 6), dpi=150)

    n = 0
    for i in range(2):
        for j in range(4):
            axs[i, j].imshow(anomalous_images[n].squeeze(), cmap="gray")
            axs[i, j].set_title(f"Label {labels[n]}")
            n += 1

    plt.show()


if __name__ == "__main__":
    ae_anom()
