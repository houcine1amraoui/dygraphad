import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    This dataset converts the time series 
    into training samples using sliding windows.
    Each sample is:
        Input  : past window of readings
        Target : next timestamp 
    This is self-supervised learning.
    """
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.T = data.shape[0]

    def __len__(self):
        return self.T - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        y = self.data[idx+self.window_size]

        # transpose to [N, window] only for GDN
        # but we don't do this since MTAD-GAT expects [W, N]
        # and we modify GDN code to do transpose
        # x = torch.tensor(x.T, dtype=torch.float32)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y