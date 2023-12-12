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
        device="cuda",
        device_ids=None
        ):
        super(MMMM, self).__init__()
        self.eeg_encoder = eeg_encoder
        self.branches = nn.ModuleDict()
        self.meta_head = meta_head.to(device)
        self.heads = nn.ModuleDict()
        self.device = device
        self.device_ids = device_ids

        self.to(device)

    def add_branch(self, name, branch):
        self.branches[name] = branch
    
    def add_head(self, name, head):
        self.heads[name] = head

    def forward(self, mode, args_dict, meta=False, staging_device="cuda:0"):
        if mode == "EEG-TEXT-BART":
            encoded_embedding = self.eeg_encoder(mode, args_dict)
            if meta:
                encoded_embedding = self.meta_head(encoded_embedding)
            else:
                encoded_embedding = self.heads[mode](encoded_embedding)
            args_dict["input_data_batch"] = encoded_embedding
            out = self.branches[mode](mode,args_dict,staging_device)
            return out
        elif mode == "EEG-IMG-BRAIN2IMAGE":
            encoded_embedding = self.eeg_encoder(mode, args_dict)
            if meta:
                encoded_embedding = self.meta_head(encoded_embedding)
            else:
                encoded_embedding = self.heads[mode](encoded_embedding)
            args_dict["input_data_batch"] = encoded_embedding
            out = self.branches[mode](mode, args_dict, staging_device)
            return out