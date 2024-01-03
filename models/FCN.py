import torch
import torch.nn as nn
import torch.nn.functional as F

# Fully Connected Network
# Used as task head, adapter, projection, etc.
class FCN(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024, hidden_dim=1024, num_layers=2, device=None, dtype=torch.float32):
        """
        Fully Connected Network (FCN) model.

        Args:
            input_dim (int): Dimension of the input features. Default is 512.
            output_dim (int): Dimension of the output features. Default is 1024.
            hidden_dim (int): Dimension of the hidden layers. Default is 1024.
            num_layers (int): Number of hidden layers. Default is 2.
            device (str): Device to use for computation. Default is "cuda".
        """
        super(FCN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.device = device
        if not device == None:
            self.to(device)
        self.dtype = dtype
        self.to(dtype=dtype)

    def forward(self, x, staging_device="cuda:0"):
        """
        Forward pass of the FCN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            staging_device (str): Device to use for intermediate computations. Default is "cuda:0".

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.activation(self.output_layer(x))
        return x