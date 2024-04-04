import torch
import torch.nn as nn
import torch.optim as optim


class Classifier(nn.Module):
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


class VerificationNet:
    def __init__(
        self, force_learn: bool = False, file_name: str = "./models/verification_model"
    ):
        self.force_relearn = force_learn
        self.file_name = file_name

        self.model = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        self.done_training = self.load()

    def load(self):
        try:
            self.model.load_state_dict(torch.load(f=self.file_name))
            return True
        except:
            print(
                "Could not read weights for verification_net from file. Must retrain..."
            )
            return False


if __name__ == "__main__":
    print("VerificationNet is not implemented in PyTorch.")
