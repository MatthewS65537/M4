import torch
import torch.nn as nn
import torch.nn.functional as F

class TextHead(nn.Module):
    def __init__(self, input_dim=840, output_dim=1024, hidden_dim=1024, num_layers=2, device="cuda:0"):
        super(TextHead, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.device = device

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.activation(self.output_layer(x))
        return x

class ImgHead(nn.Module):
    def __init__(self, input_dim=500, output_dim=1024, hidden_dim=1024, num_layers=2, device="cuda:0"):
        super(ImgHead, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.device = device

    def forward(self, x, pool=True):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.activation(self.output_layer(x))
        if not pool:
            return x
        sz = x.shape
        # Pool all in single batch together
        # Here, each batch has 128 signals corresponding to 128 channels
        pooler = torch.zeros(sz[:-2] + tuple([sz[-1]])).to(self.device)
        # Iterate through items in batch
        for i in range(sz[-2]):
            pooler += x[:,i,:]
        pooler /= sz[-2]
        return pooler

class EEGEncoder(nn.Module):
    def __init__(self, txt_in_feat=840, img_in_feat=500, enc_feat=1024, dec_emb_sz=768, enc_nhead=8, enc_dim_ff=2048, num_enc_layers=8, device="cuda:0"):
        super(EEGEncoder, self).__init__()

        self.device = device
        
        self.txt_head = TextHead(input_dim=txt_in_feat, output_dim=enc_feat, device=device)
        self.img_head = ImgHead(input_dim=img_in_feat, output_dim=enc_feat, device=device)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=enc_feat, nhead=enc_nhead, dim_feedforward=enc_dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers)
        self.fc_proj = nn.Linear(enc_feat, dec_emb_sz)

    def forward(self, mode, input_data_batch=None, input_masks_batch=None, input_masks_invert=None, pool_img_head=True):
        if mode == "TXT":
            encoded_embedding = self.txt_head(input_data_batch)
        elif mode == "IMG":
            encoded_embedding = self.img_head(input_data_batch, pool_img_head)

        encoded_embedding = self.encoder(encoded_embedding, src_key_padding_mask = input_masks_invert)
        encoded_embedding = F.relu(self.fc_proj(encoded_embedding))
        return encoded_embedding