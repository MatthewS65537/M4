import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./models")

from FCN import *
from EEGEncoder import *

class Branch(nn.Module):
    def __init__(self, head, body, device="cuda", device_ids=None):
        """
        Initializes a Branch module.

        Args:
            head (nn.Module): The head module.
            body (nn.Module): The body module.
            device (str, optional): The device to use. Defaults to "cuda".
            device_ids (list, optional): List of device IDs for data parallelism. Defaults to None.
        """
        super(Branch, self).__init__()
        self.head = head
        self.body = body
        self.to(device)
        self.device = device
        self.device_ids = device_ids

    def forward(self, mode, args_dict, staging_device="cuda:0"):
        """
        Forward pass of the Branch module.

        Args:
            mode (str): The mode of operation.
            args_dict (dict): Dictionary of input arguments.
            staging_device (str, optional): The device to use for staging. Defaults to "cuda:0".

        Returns:
            The output of the forward pass.
        """
        if mode == "EEG-TEXT-BART":
            input_data_batch = args_dict["input_data_batch"]
            input_masks_batch = args_dict["input_masks_batch"]
            target_ids_batch = args_dict["target_ids_batch"]
            encoded_embedding = self.head(input_data_batch)
            # Body is pretrained BART
            out = self.body(
                inputs_embeds = encoded_embedding,
                attention_mask = input_masks_batch,
                return_dict = True,
                labels = target_ids_batch
                )
            return out
        elif mode == "EEG-IMG-DIFFUSION":
            input_data_batch = args_dict["input_data_batch"]
            encoded_embedding = self.head(input_data_batch)
            # Body is DiffusionHead()
            out = self.body(args_dict)
            return out
        elif mode == "EEG-IMG-BRAIN2IMAGE-CLASSIFICATION":
            input_data_batch = args_dict["input_data_batch"]
            # Body is ClassificationHead()
            encoded_embedding = self.head(input_data_batch)
            out = self.body(encoded_embedding)
            return out # Return predicted probabilities for each class
        elif mode == "EEG-TEXT-BART-SENTIMENT": # Same as EEG-IMG-CLASSIFICATION
            input_data_batch = args_dict["input_data_batch"]
            # Body is ClassificationHead()
            encoded_embedding = self.head(input_data_batch)
            out = self.body(encoded_embedding)
            return out # Return predicted probabilities for each class