import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stacked_mnist import StackedMNISTData
from torch.utils.data import DataLoader, TensorDataset


class VerificationNet:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(p=0.5),
            )

            self.flatten = nn.Flatten()

            self.decoder = nn.Sequential(
                nn.Linear(in_features=28 * 28, out_features=128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=128, out_features=10),
                nn.Softmax(dim=1),
            )

        def forward(self, x):
            z = self.encoder(x)
            z = self.flatten(z)
            return self.decoder(z)

    def __init__(
        self, force_learn: bool = False, file_name: str = "./models/verification_model"
    ):
        self.force_relearn = force_learn
        self.file_name = file_name

        self.model = self.Model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        self.done_training = self.load()

    def load(self):
        try:
            self.model.load_state_dict(torch.load(f=self.file_name))
            return True
        except:  # noqa: E722
            print(
                "Could not read weights for verification_net from file. Must retrain..."
            )
            return False

    def train(self, generator: StackedMNISTData, epochs: int = 10):
        self.done_training = self.load()

        if self.force_relearn or not self.done_training:
            x_train, y_train = generator.get_full_data_set(training=True)
            x_test, y_test = generator.get_full_data_set(training=False)

            x_train = torch.tensor(x_train[:, :, :, [0]], dtype=torch.float32)
            y_train = torch.tensor((y_train % 10).astype(np.int), dtype=torch.long)
            y_train = F.one_hot(y_train, num_classes=10)

            x_test = torch.tensor(x_test[:, :, :, [0]], dtype=torch.float32)
            y_test = torch.tensor((y_test % 10).astype(np.int), dtype=torch.long)
            y_test = F.one_hot(y_test, num_classes=10)

            # Create PyTorch datasets and dataloaders
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

            for epoch in range(epochs):
                self.model.train()

                for X, y in train_loader:
                    self.optimizer.zero_grad()
                    pred = self.model(X)
                    loss = self.criterion(pred, torch.argmax(y, dim=1))
                    loss.backward()
                    self.optimizer.step()

                self.model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for X, y in test_loader:
                        pred = self.model(X)
                        test_loss += self.criterion(pred, torch.argmax(y, dim=1)).item()
                        pred = pred.argmax(dim=1, keepdim=True)
                        correct += (
                            pred.eq(torch.argmax(y, dim=1).view_as(pred)).sum().item()
                        )

                test_loss /= len(test_loader.dataset)
                accuracy = 100.0 * correct / len(test_loader.dataset)
                print(
                    f"Epoch: {epoch +1 }, Test Loss: {test_loss}, Accuracy: {accuracy}%"
                )

            torch.save(self.model.state_dict(), f=self.file_name)


if __name__ == "__main__":
    print("VerificationNet is not implemented in PyTorch.")
