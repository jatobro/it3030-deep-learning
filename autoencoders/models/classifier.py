import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, image_depth=1, n_classes=10):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(image_depth, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
