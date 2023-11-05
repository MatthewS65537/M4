import torch
import torch.nn as nn
import torch.nn.functional as F

# Fully Connected Network
# Used as task head, adapter, projection, etc.
class FCN(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024, hidden_dim=1024, num_layers=2, device="cuda"):
        super(FCN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.device = device

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.activation(self.output_layer(x))
        return x