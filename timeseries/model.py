import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Convolutional(nn.Module):
    def __init__(self, tw):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv1d(
                in_channels=tw, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2, 2),
        )

        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_stack(x)

        return x


class LSTM(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(
            device=device, input_size=1, hidden_size=64, num_layers=1, batch_first=True
        )
        self.linear1 = nn.Linear(64 * 100, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        hidden_state = torch.zeros(1, x.shape[0], 64).to(self.device)
        cell_state = torch.zeros(1, x.shape[0], 64).to(self.device)

        hidden = (hidden_state, cell_state)

        x, _ = self.lstm(x, hidden)
        x = x.contiguous().view(x.shape[0], -1)
        x = torch.relu(self.linear1(x))
        return self.linear2(x)
