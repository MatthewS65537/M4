import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskHead(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024, hidden_dim=1024, num_layers=2, device="cuda"):
        super(TaskHead, self).__init__()
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

class EEGEncoder(nn.Module):
    def __init__(self, enc_feat=1024, dec_emb_sz=768, enc_nhead=8, enc_dim_ff=2048, num_enc_layers=8, device="cuda"):
        super(EEGEncoder, self).__init__()

        self.device = device

        self.task_heads = {}

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=enc_feat, nhead=enc_nhead, dim_feedforward=enc_dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers)
        self.fc_proj = nn.Linear(enc_feat, dec_emb_sz)

    def add_task(self, task_name, task_head):
        self.task_heads[task_name] = task_head

    def forward(self, mode, input_data_batch=None, input_masks_batch=None, input_masks_invert=None, pool_result=False):
        encoded_embedding = self.task_heads[mode](input_data_batch)
        encoded_embedding = self.encoder(encoded_embedding, src_key_padding_mask = input_masks_invert)
        encoded_embedding = F.relu(self.fc_proj(encoded_embedding))
        
        if pool_result:
            pooler = torch.zeros(encoded_embedding.shape[0:1]+encoded_embedding.shape[-1:]).to(device)
            for i in range(encoded_embedding.shape[0]):
              for j in range(encoded_embedding.shape[-2]):
                  pooler[i] += encoded_embedding[i][j][:]
              pooler[i] /= encoded_embedding.shape[-2]
            return pooler
        return encoded_embedding