import torch
from torch.utils.data import Dataset
from data_loader import load_data

class FaultDataset(Dataset):
    def __init__(self, folder="simulation_results", timesteps=651):
        X, Z, y, encoder = load_data(folder, timesteps)
        self.X = torch.tensor(X, dtype=torch.float32)   # (samples, timesteps, features)
        self.Z = torch.tensor(Z, dtype=torch.float32)   # (samples,)
        self.y = torch.tensor(y, dtype=torch.long)
        self.encoder = encoder

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx], self.y[idx]