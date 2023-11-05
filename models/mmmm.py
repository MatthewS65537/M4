import torch
import torch.nn as nn
import torch.nn.functional as F

class MMMM(nn.Module):
    def __init__(self, eeg_encoder):
        self.eeg_encoder = eeg_encoder
        self.branches = {}

    def add_branch(self, branch_name, branch):
        self.branches[branch_name] = branch

    def forward(self, mode, args_dict):
        args_dict