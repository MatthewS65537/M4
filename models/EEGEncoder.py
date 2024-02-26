import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./models")

from FCN import *
from PositionalEncoder import PositionalEncoder
# Main EEGEncoder() Class
class EEGEncoder(nn.Module):
    """
    Encoder module for EEG data.

    Args:
        enc_feat (int): The number of encoder features.
        dec_emb_sz (int): The size of the decoder embedding.
        enc_nhead (int): The number of attention heads in the encoder.
        enc_dim_ff (int): The dimension of the feedforward network in the encoder.
        num_enc_layers (int): The number of encoder layers.
        device (str): The device to use for computation (default: "cuda").
        device_ids (list): List of device IDs for multi-GPU training (default: None).
    """

    def __init__(self, enc_feat=1024, dec_emb_sz=768, enc_nhead=8, enc_dim_ff=2048, num_enc_layers=8, device=None, device_ids=None, dtype=torch.float32, pos_emb = True):
        super(EEGEncoder, self).__init__()

        self.device = device
        self.heads = nn.ModuleDict()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=enc_feat, nhead=enc_nhead, dim_feedforward=enc_dim_ff, batch_first=True).to(dtype=dtype)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers).to(dtype=dtype)
        self.fc_proj = nn.Linear(enc_feat, dec_emb_sz)
        self.device_ids = device_ids
        self.dtype = dtype
        self.device = device
        self.pos_emb = pos_emb
        if self.pos_emb:
            self.pos_encoder = PositionalEncoder(dim=enc_feat, dropout=0.1, max_len=5000).to(dtype=dtype)
        if not device == None:
            self.to(device)

        self.to(dtype=dtype)

        
        
    def add_head(self, name, head):
        """
        Add a head module to the encoder.

        Args:
            name (str): The name of the head.
            head (nn.Module): The head module to add.
        """
        self.heads[name] = head.to(dtype=self.dtype)

    def forward(self, mode, args_dict, staging_device="cuda:0", debug=False):
        """
        Forward pass of the encoder.

        Args:
            mode (str): The mode of the encoder.
            args_dict (dict): Dictionary containing input data and other arguments.
            staging_device (str): The device to use for staging (default: "cuda:0").

        Returns:
            torch.Tensor: The encoded embedding.
        """
        if mode == "EEG-TEXT-BART":
            input_data_batch = args_dict["input_data_batch"]
            input_masks_invert = args_dict["input_masks_invert"]
            encoded_embedding = self.heads[mode](input_data_batch)
            if debug:
                print("ENCODER HEAD GOOD")
            encoded_embedding = self.encoder(encoded_embedding, src_key_padding_mask=input_masks_invert)
            if self.pos_emb:
                input_masks = args_dict["input_masks_batch"]
                encoded_embedding = self.pos_encoder(encoded_embedding, mask=input_masks)
            if debug:
                print("ENCODER ENCODER GOOD")
            encoded_embedding = self.fc_proj(encoded_embedding)
#             encoded_embedding = F.gelu(encoded_embedding)
            if debug:
                print("ENCODER FC GOOD")
            pool_result = args_dict["pool_result"]
            if pool_result:
                encoded_embedding = torch.mean(encoded_embedding, dim=1)
            return encoded_embedding

        elif mode == "EEG-IMG-BRAIN2IMAGE":
            input_data_batch = args_dict["input_data_batch"]
            input_masks_invert = args_dict["input_masks_invert"]
            encoded_embedding = self.heads[mode](input_data_batch)
            encoded_embedding = self.encoder(encoded_embedding, src_key_padding_mask=input_masks_invert)
            if self.pos_emb:
                input_masks = args_dict["input_masks_batch"]
                encoded_embedding = self.pos_encoder(encoded_embedding, mask=input_masks)
            encoded_embedding = self.fc_proj(encoded_embedding)
#             encoded_embedding = F.gelu(encoded_embedding)
            pool_result = args_dict["pool_result"]
            if pool_result:
                encoded_embedding = torch.mean(encoded_embedding, dim=1)
            return encoded_embedding
