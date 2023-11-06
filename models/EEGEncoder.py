import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./models")

from FCN import *

class EEGEncoder(nn.Module):
    def __init__(self, enc_feat=1024, dec_emb_sz=768, enc_nhead=8, enc_dim_ff=2048, num_enc_layers=8, device="cuda", device_ids=None):
        super(EEGEncoder, self).__init__()
        self.device = device
        self.heads = nn.ModuleDict()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=enc_feat, nhead=enc_nhead, dim_feedforward=enc_dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers)
        self.fc_proj = nn.Linear(enc_feat, dec_emb_sz)
        self.device_ids = device_ids
        self.to(device)
        
    def add_head(self, name, head):
        self.heads[name] = head
#         if self.device_ids == None:
#             self.heads[name] = head.to(self.device)
#         else:
#             self.heads[name] = nn.DataParallel(head.to(self.device),device_ids=self.device_ids)

    def forward(self, mode, args_dict, staging_device="cuda:0"):
        if mode == "EEG-TEXT-BART":
            input_data_batch = args_dict["input_data_batch"]
            input_masks_invert = args_dict["input_masks_invert"]
            encoded_embedding = self.heads[mode](input_data_batch)
            encoded_embedding = self.encoder(encoded_embedding,src_key_padding_mask=input_masks_invert)
            encoded_embedding = F.relu(self.fc_proj(encoded_embedding))
            return encoded_embedding

        elif mode == "EEG-IMG-BRAIN2IMAGE":
            input_data_batch = args_dict["input_data_batch"]
            encoded_embedding = self.heads[mode](input_data_batch)
            pool_result = args_dict["pool_result"]
            if pool_result:
                pooler = torch.zeros(encoded_embedding.shape[0:1]+encoded_embedding.shape[-1:]).to(device)
                for i in range(encoded_embedding.shape[0]):
                  for j in range(encoded_embedding.shape[-2]):
                      pooler[i] += encoded_embedding[i][j][:]
                  pooler[i] /= encoded_embedding.shape[-2]
                return pooler
