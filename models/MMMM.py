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
        emb_unet=None,
        device=None,
        device_ids=None,
        dtype=torch.float32
    ):
        super(MMMM, self).__init__()
        self.dtype = dtype
        self.BART_text_encoder = None if BART_text_encoder == None else BART_text_encoder.to(dtype=dtype)
        self.CLIP_text_encoder = None if CLIP_text_encoder == None else CLIP_text_encoder.to(dtype=dtype)
        self.image_encoder = None if image_encoder == None else image_encoder.to(dtype=dtype)
        self.eeg_encoder = eeg_encoder.to(dtype=dtype)
        self.dual_encoder = None if dual_encoder == None else dual_encoder.to(dtype=dtype)
        self.fusion_encoder = None if fusion_encoder == None else fusion_encoder.to(dtype=dtype)
        self.emb_unet = None if emb_unet == None else fusion_encoder.to(dtype=dtype)
        self.branches = nn.ModuleDict()
        self.meta_head = meta_head.to(dtype=dtype)
        self.heads = nn.ModuleDict()
        self.device = device
        self.device_ids = device_ids
        
        if not device == None:
            self.to(device)

    def add_branch(self, name, branch):
        """
        Adds a branch to the model.

        Args:
            name (str): The name of the branch.
            branch (nn.Module): The branch module to be added.

        """
        self.branches[name] = branch.to(dtype=self.dtype)
    
    def add_head(self, name, head):
        """
        Adds a head to the model.

        Args:
            name (str): The name of the head.
            head (nn.Module): The head module to be added.

        """
        self.heads[name] = head.to(dtype=self.dtype)

    def forward(self, mode, args_dict, meta=False, staging_device="cuda:0", debug=False):
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
        if mode == "PRETRAIN-EEG-TEXT-CLIP-MATCHING":
            encoded_embedding = self.eeg_encoder("EEG-TEXT-BART", args_dict, staging_device, debug)
            use_unet = args_dict["use_unet"] if "use_unet" in args_dict else False
            if use_unet:
                encoded_embedding = self.meta_head(encoded_embedding)
            encoded_embedding = self.emb_unet(encoded_embedding.reshape(encoded_embedding.shape[0], encoded_embedding.shape[2], encoded_embedding.shape[1]))
            return encoded_embedding
        elif mode == "PRETRAIN-EEG-IMG-CLIP-MATCHING":
            encoded_embedding = self.eeg_encoder("EEG-IMG-BRAIN2IMAGE", args_dict, staging_device, debug)
            if use_unet:
                encoded_embedding = self.emb_unet(encoded_embedding.reshape(encoded_embedding.shape[0], encoded_embedding.shape[2], encoded_embedding.shape[1]))
            encoded_embedding = encoded_embedding.reshape(encoded_embedding.shape[0], encoded_embedding.shape[2], encoded_embedding.shape[1])
            return encoded_embedding
        elif mode == "EEG-TEXT-BART":
            encoded_embedding = self.eeg_encoder(mode, args_dict, staging_device, debug)
            if debug:
                print("ENCODER GOOD")
            if meta:
                encoded_embedding = self.meta_head(encoded_embedding)
            else:
                encoded_embedding = self.heads[mode](encoded_embedding)
            if debug:
                print("HEAD GOOD")
            args_dict["input_data_batch"] = encoded_embedding
            out = self.branches[mode](mode, args_dict, staging_device, debug)
            if debug:
                print("BRANCH GOOD")
            return out
        elif mode == "EEG-IMG-DIFFUSION":
            if "train" in args_dict and args_dict["train"]:
                encoded_embedding = self.eeg_encoder("EEG-IMG-BRAIN2IMAGE", args_dict)
                if debug:
                    print("ENCODER GOOD")
                if meta:
                    encoded_embedding = self.meta_head(encoded_embedding)
                else:
                    encoded_embedding = self.heads[mode](encoded_embedding)
                if debug:
                    print("HEAD GOOD")
                args_dict["input_data_batch"] = encoded_embedding
                out = self.branches[mode](mode, args_dict, staging_device, debug)
                if debug:
                    print("BRANCH GOOD")
                return out
            else:   
                encoded_embedding = self.eeg_encoder("EEG-IMG-BRAIN2IMAGE", args_dict)
                if debug:
                    print("ENCODER GOOD")
                if meta:
                    encoded_embedding = self.meta_head(encoded_embedding)
                else:
                    encoded_embedding = self.heads[mode](encoded_embedding)
                if debug:
                    print("HEAD GOOD")
                args_dict["input_data_batch"] = encoded_embedding
                out = self.branches[mode](mode, args_dict, staging_device)
                if debug:
                    print("BRANCH GOOD")
                return out
        elif mode == "EEG-IMG-BRAIN2IMAGE-CLASSIFICATION":
            encoded_embedding = self.eeg_encoder("EEG-IMG-BRAIN2IMAGE", args_dict)
            if meta:
                encoded_embedding = self.meta_head(encoded_embedding)
            else:
                encoded_embedding = self.heads[mode](encoded_embedding)
            args_dict["input_data_batch"] = encoded_embedding
            out = self.branches[mode](mode, args_dict, staging_device)
            return out
        elif mode == "EEG-TEXT-BART-SENTIMENT":
            encoded_embedding = self.eeg_encoder("EEG-TEXT-BART", args_dict)
            if meta:
                encoded_embedding = self.meta_head(encoded_embedding)
            else:
                encoded_embedding = self.heads[mode](encoded_embedding)
            args_dict["input_data_batch"] = encoded_embedding
            out = self.branches[mode](mode, args_dict, staging_device)
            return out
        else:
            print(f"Mode {mode} not found")