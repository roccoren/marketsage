import torch
import torch.nn as nn

class GRUModel_3Layers(nn.Module):
    def __init__(self, num_features, num_timesteps):
        super(GRUModel_3Layers, self).__init__()
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout_rate = 0.2
        
        self.gru1 = nn.GRU(input_size=num_features, hidden_size=self.hidden_size, num_layers=1, batch_first=False, dropout=self.dropout_rate)
        self.gru2 = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=False, dropout=self.dropout_rate)
        self.gru3 = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=False, dropout=self.dropout_rate)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_timesteps, num_features)
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out, _ = self.gru3(out)
        out = self.fc(out[-1])  # Use the output of the last time step
        return out

model_cls = GRUModel_3Layers