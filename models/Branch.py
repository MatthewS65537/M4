import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./models")

from FCN import *
from EEGEncoder import *

class Branch(nn.Module):
    def __init__(self, head, body, device="cuda"):
        super(Branch, self).__init__()
        self.head = head
        self.body = body
        self.device = device
        self.to(device)

    def forward(self, mode, args_dict):
        if mode == "EEG-TEXT-BART":
            input_data_batch = args_dict["input_data_batch"]
            input_masks_batch = args_dict["input_masks_batch"]
            target_ids_batch = args_dict["target_ids_batch"]
            encoded_embedding = self.head(input_data_batch)
            out = self.body(
                inputs_embeds = encoded_embedding,
                attention_mask = input_masks_batch,
                return_dict = True,
                labels = target_ids_batch
                )
            return out