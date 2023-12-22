import torch
import torch.nn as nn
import torch.nn.functional as F

class DualEncoder(nn.Module):
    def __init__(self, out_dim, device="cuda", device_ids=None):
        super(DualEncoder, self).__init__()
        self.device = device
        self.fc = nn.ModuleDict()
        self.out_dim = out_dim
        self.to(device)
    
    def add_mode(self, name, in_dim):
        self.fc[name] = nn.Linear(in_dim, self.out_dim)

    def forward(self, x):
        out = None
        for data_dict in x:
            if out == None:
                out = self.fc[data_dict["mode"]](data_dict["embed"])
            else:
                out += self.fc[data_dict["mode"]](data_dict["embed"])
        return out