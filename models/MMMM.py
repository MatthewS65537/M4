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
        self.device = device
        self.device_ids = device_ids

        self.to(device)

    def add_branch(self, name, branch):
        self.branches[name] = branch
#         if self.device == None:
#             self.branches[name] = branch.to(self.device)
#         else:
#             self.branches[name] = nn.DataParallel(branch.to(self.device), device_ids=self.device_ids)

    def forward(self, mode, args_dict, staging_device="cuda:0"):
        if mode == "EEG-TEXT-BART":
            encoded_embedding = self.eeg_encoder(mode, args_dict)
            encoded_embedding = self.meta_head(encoded_embedding)
            args_dict["input_data_batch"] = encoded_embedding
            out = self.branches[mode](mode,args_dict,staging_device)
            return out