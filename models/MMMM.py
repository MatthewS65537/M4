import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./models")

from FCN import *
from Branch import *
from DualEncoder import *

class MMMM(nn.Module):
    """
    MMMM class represents a neural network model that combines EEG and other modalities for multi-modal learning.

    Args:
        eeg_encoder (nn.Module): The EEG encoder module.
        meta_head (nn.Module, optional): The meta head module. Defaults to FCN with input_dim=768, output_dim=768, and num_layers=4.
        device (str, optional): The device to use for computation. Defaults to "cuda".
        device_ids (list, optional): The list of device IDs to use for parallel computation. Defaults to None.

    Attributes:
        eeg_encoder (nn.Module): The EEG encoder module.
        branches (nn.ModuleDict): The dictionary to store different branches of the model.
        meta_head (nn.Module): The meta head module.
        heads (nn.ModuleDict): The dictionary to store different heads of the model.
        device (str): The device used for computation.
        device_ids (list): The list of device IDs used for parallel computation.

    Methods:
        add_branch(name, branch): Adds a branch to the model.
        add_head(name, head): Adds a head to the model.
        forward(mode, args_dict, meta=False, staging_device="cuda:0"): Performs forward pass through the model.

    """

    def __init__(
        self,
        eeg_encoder,
        dual_encoder,
        BART_text_encoder,
        CLIP_text_encoder,
        image_encoder,
        fusion_encoder,
        meta_head=FCN(
            input_dim=768,
            output_dim=768,
            num_layers=4,
        ),
        device="cuda",
        device_ids=None
    ):
        super(MMMM, self).__init__()
        self.BART_text_encoder = BART_text_encoder
        self.CLIP_text_encoder = CLIP_text_encoder
        self.image_encoder = image_encoder
        self.eeg_encoder = eeg_encoder
        self.dual_encoder = dual_encoder
        self.fusion_encoder = fusion_encoder
        self.branches = nn.ModuleDict()
        self.meta_head = meta_head.to(device)
        self.heads = nn.ModuleDict()
        self.device = device
        self.device_ids = device_ids

        self.to(device)

    def add_branch(self, name, branch):
        """
        Adds a branch to the model.

        Args:
            name (str): The name of the branch.
            branch (nn.Module): The branch module to be added.

        """
        self.branches[name] = branch
    
    def add_head(self, name, head):
        """
        Adds a head to the model.

        Args:
            name (str): The name of the head.
            head (nn.Module): The head module to be added.

        """
        self.heads[name] = head

    def forward(self, mode, args_dict, meta=False, staging_device="cuda:0"):
        """
        Performs forward pass through the model.

        Args:
            mode (str): The mode of the model.
            args_dict (dict): The dictionary containing input data.
            meta (bool, optional): Whether to use the meta head. Defaults to False.
            staging_device (str, optional): The device to use for staging. Defaults to "cuda:0".

        Returns:
            The output of the forward pass.

        """
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
            if "train" in args_dict and args_dict["train"]:
                encoded_embedding = self.eeg_encoder(mode, args_dict)
                if meta:
                    encoded_embedding = self.meta_head(encoded_embedding)
                else:
                    encoded_embedding = self.heads[mode](encoded_embedding)
                return encoded_embedding
            else:   
                encoded_embedding = self.eeg_encoder(mode, args_dict)
                if meta:
                    encoded_embedding = self.meta_head(encoded_embedding)
                else:
                    encoded_embedding = self.heads[mode](encoded_embedding)
                args_dict["input_data_batch"] = encoded_embedding
                out = self.branches[mode](mode, args_dict, staging_device)
                return out
        elif mode == "EEG-IMG-CLASSIFICATION":
            encoded_embedding = self.eeg_encoder("EEG-IMG-BRAIN2IMAGE", args_dict)
            if meta:
                encoded_embedding = self.meta_head(encoded_embedding)
            else:
                encoded_embedding = self.heads[mode](encoded_embedding)
            args_dict["input_data_batch"] = encoded_embedding
            out = self.branches[mode](mode, args_dict, staging_device)
            return encoded_embedding
        elif mode == "EEG-SENTIMENT-ANALYSIS":
            encoded_embedding = self.eeg_encoder("EEG-TEXT-BART", args_dict)
            if meta:
                encoded_embedding = self.meta_head(encoded_embedding)
            else:
                encoded_embedding = self.heads[mode](encoded_embedding)
            args_dict["input_data_batch"] = encoded_embedding
            out = self.branches[mode](mode, args_dict, staging_device)
            return encoded_embedding