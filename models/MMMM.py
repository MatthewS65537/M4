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
            ),
        device="cuda"
        ):
        super(MMMM, self).__init__()
        self.eeg_encoder = eeg_encoder
        self.branches = {}
        self.meta_head = meta_head.to(device)
        self.device = device

        self.to(device)

    def add_branch(self, name, branch):
        self.branches[name] = branch.to(self.device)

    def forward(self, mode, args_dict):
        if mode == "EEG-TEXT-BART":
            encoded_embedding = self.eeg_encoder(mode, args_dict)
            encoded_embedding = self.meta_head(encoded_embedding)
            args_dict["input_data_batch"] = encoded_embedding
            out = self.branches["EEG-TEXT-BART"](mode,args_dict)
            return out