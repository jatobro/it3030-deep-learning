import matplotlib.pyplot as plt
import numpy as np
import torch
from models.standard import StandardAutoencoder
from stacked_mnist import DataMode, StackedMNISTData
from torch.utils.data import DataLoader, TensorDataset


def ae_stacked_gen():
    model = StandardAutoencoder(in_channels=3)
    model.load_state_dict(torch.load("trained_models/ae_stacked.pth"))

    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=9)
    test_data = gen.get_full_data_set(training=False)

    X_test, y_test = test_data

    X_test = torch.from_numpy(X_test.astype(np.float32)).permute(0, 3, 1, 2)
    y_test = torch.from_numpy(y_test.astype(np.float32))

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32)

    images, _ = next(iter(test_loader))
    latent = model.encoder(images)

    mean = latent.mean(dim=0)
    std = (latent - mean).pow(2).mean(dim=0).sqrt()

    z = torch.randn(32, 16) * std + mean

    model.eval()
    with torch.no_grad():
        generated_images = model.decode(z)

    with torch.no_grad():
        fig, axs = plt.subplots(2, 5, figsize=(10, 4), dpi=150)

        n = 0
        for i in range(2):
            for j in range(5):
                axs[i, j].imshow(generated_images[n].permute(1, 2, 0))
                axs[i, j].axis("off")
                n += 1

        plt.show()


if __name__ == "__main__":
    ae_stacked_gen()
