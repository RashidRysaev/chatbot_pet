import torch
import torch.nn as nn
from torch.utils.data import Dataset

from training import x_train, y_train


class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = x_train
        self.y_data = y_train
        self.n_samples = len(x_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
