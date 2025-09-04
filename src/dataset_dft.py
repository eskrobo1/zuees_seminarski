import torch
from torch.utils.data import Dataset
from data_loader_dft import load_dft_data

class FaultDFTDataset(Dataset):
    def __init__(self, folder="simulation_results", timesteps=651, fs=1000.0):
        X, Z, y, encoder = load_dft_data(folder, timesteps, fs)
        self.X = torch.tensor(X, dtype=torch.float32)   # (samples, freq_bins)
        self.Z = torch.tensor(Z, dtype=torch.float32).unsqueeze(1)  # (samples,1)
        self.y = torch.tensor(y, dtype=torch.long)
        self.encoder = encoder

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx], self.y[idx]
