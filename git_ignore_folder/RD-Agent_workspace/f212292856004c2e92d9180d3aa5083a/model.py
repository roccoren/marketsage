import torch
import torch.nn as nn

class GRUModel_3Layers(nn.Module):
    def __init__(self, num_features, num_timesteps):
        super(GRUModel_3Layers, self).__init__()
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout_rate = 0.2
        
        # Define the GRU layers with batch_first=True and using num_layers
        self.gru = nn.GRU(input_size=num_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # x is expected to be in shape (batch_size, num_timesteps, num_features)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out

model_cls = GRUModel_3Layers