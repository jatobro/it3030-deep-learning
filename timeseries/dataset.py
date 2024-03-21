import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDatasetLSTM(Dataset):
    def __init__(self, df, time_window: int):
        self.df = df
        self.time_window = time_window

    def __len__(self):
        return len(self.df) - self.time_window

    def __getitem__(self, idx):
        sequence = torch.tensor(
            self.df[idx : idx + self.time_window].values.astype(np.float32)
        )
        target = torch.tensor(
            self.df[idx + self.time_window : idx + self.time_window + 1].values.astype(
                np.float32
            )
        )

        return sequence, target


class SequenceDatasetCNN(Dataset):
    def __init__(self, df, time_window: int, padding: int):
        self.df = df
        self.time_window = time_window
        self.padding = padding

    def __len__(self):
        return len(self.df) - self.time_window

    def __getitem__(self, idx):
        sequence = torch.tensor(
            self.df[idx : idx + self.time_window].values.astype(np.float32)
        )
        target = torch.tensor(
            self.df[idx + self.time_window : idx + self.time_window + self.padding]
            .values[:, 0]
            .astype(np.float32)
        )

        return sequence, target
