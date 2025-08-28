import torch
import torch.nn as nn

class FaultClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FaultClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.net(x)
