from torch import nn, optim


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
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


class VerificationNet:
    def __init__(self):
        self.model = ClassificationModel()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)


if __name__ == "__main__":
    print("VerificationNet is not implemented in PyTorch.")
