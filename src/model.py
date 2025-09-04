import torch
import torch.nn as nn

class CNN_RNN_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CNN_RNN_Classifier, self).__init__()

        # CNN dio
        '''
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        '''
        # RNN dio
        #self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
       
        # Klasifikator (hidden_dim + 1 jer dodajemo Z_fault)
        self.fc = nn.Linear(hidden_dim + 1, num_classes)

    def forward(self, x, zfault):
        # x shape: (batch, timesteps, features) - direktno u LSTM
        out, (h, c) = self.lstm(x)
        h = h[-1]                    # zadnji hidden state
        # dodaj Z_fault
        zfault = zfault.unsqueeze(1)  # (batch,1)
        h = torch.cat([h, zfault], dim=1)
        return self.fc(h)