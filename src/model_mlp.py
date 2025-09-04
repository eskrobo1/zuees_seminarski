import torch
import torch.nn as nn

class MLP_DFT_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_DFT_Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 128),  # +1 zbog Zfault
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, zfault):
        # x: (batch, freq_bins), zfault: (batch,1)
        inp = torch.cat([x, zfault], dim=1)
        return self.net(inp)
