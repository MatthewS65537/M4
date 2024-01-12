import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=512, output_dim=10, hidden_dim=1024, num_layers=2, device=None, dtype=torch.float32, dropout=0.25, activation=None):
        """
        Classification Neural Network model.

        Args:
            input_dim (int): Dimension of the input features. Default is 512.
            output_dim (int): Number of classes for classification. Default is 10.
            hidden_dim (int): Dimension of the hidden layers. Default is 1024.
            num_layers (int): Number of hidden layers. Default is 2.
            device (str): Device to use for computation. Default is "cuda".
        """
        super(ClassificationHead, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)  # Add softmax activation
        self.device = device
        self.dtype = dtype
        if not device == None:
            self.to(device)
        self.to(dtype)
        self.dropout = dropout
        self.activation = None if activation == None else activation

    def forward(self, x, temperature, staging_device="cuda:0"):
        """
        Forward pass of the Classification Neural Network model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            staging_device (str): Device to use for intermediate computations. Default is "cuda:0".

        Returns:
            torch.Tensor: Class probabilities tensor of shape (batch_size, output_dim).
        """
        for layer in self.hidden_layers:
            if not self.dropout == None:
                x = nn.Dropout(p=self.dropout)(x)
            layer = layer(x)
            if not self.activation == None:
                x = self.activation(x)
        x = self.softmax(self.output_layer(x)/temperature)  # Apply softmax activation
        return x