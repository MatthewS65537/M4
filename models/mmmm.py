import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./models")

from FCN import *
from EEGEncoder import *

class Branch(nn.Module):
    def __init__(self, head, body, device="cuda"):
        self.head = head
        self.body = body
        self.device = device

    def forward(self, mode, args_dict):
        if mode == "EEG-TEXT-BART":
            input_data_batch = args_dict["input_data_batch"]
            input_masks_batch = args_dict["input_masks_batch"]
            target_ids_batch_converted = ["target_ids_batch_converted"]
            out = self.head(input_data_batch)
            out = self.body(
                inputs_embeds = encoded_embedding,
                attention_mask = input_masks_batch,
                return_dict = True,
                labels = target_ids_batch_converted
                )
            return out

class MMMM(nn.Module):
    def __init__(self, eeg_encoder, device="cuda"):
        super(MMMM, self).__init__()
        self.eeg_encoder = eeg_encoder
        self.branches = {}
        self.device = device

    def add_branch(self, name, branch):
        self.branches[name] = branch

    def forward(self, mode, args_dict):
        if mode == "EEG-TEXT-BART":
            encoded_embedding = self.eeg_encoder(mode, args_dict)
            out = self.branch["EEG-TEXT-BART"](args_dict)
            return out