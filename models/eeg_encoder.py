import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGEncoder(nn.Module):
    def __init__(self, txt_in_feat=840, img_in_feat=500, enc_feat=1024, dec_emb_sz=768, enc_nhead=8, enc_dim_ff=2048, num_enc_layers=8):
        super(EEGEncoder, self).__init__()

        self.txt_fc_head = nn.Linear(txt_in_feat, enc_in_feat)
        self.img_fc_head = nn.Linear(img_in_feat, enc_in_feat)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=enc_feat, nhead=enc_nhead, dim_feedforward=enc_dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers)
        self.fc_proj = nn.Linear(enc_feat, decoder_embedding_size)

    def forward(self, mode, input_embeddings_batch=None, input_masks_batch=None, input_masks_invert=None):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        try:
            if mode == "IMG":
                encoded_embedding = self.img_fc_head(input_embeddings_batch)
                encoded_embedding = self.encoder(input_embeddings_batch, src_key_padding_mask = input_masks_invert)
                encoded_embedding = F.relu(self.fc_proj(encoded_embedding))
            elif mode == "TXT":
                encoded_embedding = self.txt_fc_head(input_embeddings_batch)
                encoded_embedding = self.encoder(input_embeddings_batch, src_key_padding_mask = input_masks_invert)
                encoded_embedding = F.relu(self.fc_proj(encoded_embedding))
            return encoded_embedding
        except:
            return None