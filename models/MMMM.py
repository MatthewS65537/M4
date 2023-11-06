import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./models")

from FCN import *
from Branch import *

class MMMM(nn.Module):
    def __init__(
        self,
        eeg_encoder,
        meta_head=FCN(
            input_dim=768,
            output_dim=768,
            num_layers=4,
            device=device
            ),
        device="cuda"
        ):
        super(MMMM, self).__init__()
        self.eeg_encoder = eeg_encoder
        self.branches = {}
        self.meta_head = meta_head
        self.device = device

    def add_branch(self, name, branch):
        self.branches[name] = branch

    def forward(self, mode, args_dict):
        if mode == "EEG-TEXT-BART":
            encoded_embedding = self.eeg_encoder(mode, args_dict)
            encoded_embedding = self.meta_head(encoded_embedding)
            args_dict["input_data_batch"] = 
            out = self.branch["EEG-TEXT-BART"](args_dict)
            return out