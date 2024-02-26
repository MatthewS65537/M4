import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)
        # Precompute and store positional encoding for a large enough sequence length
        self.register_buffer('positional_encodings', self._generate_positional_encodings(max_len, dim))

    def _generate_positional_encodings(self, pos_len, dim):
        assert dim % 2 == 0, "Dimension must be an even number!"
        position_emb = torch.zeros(pos_len, dim, dtype=torch.float)

        # i matrix
        i_matrix = torch.arange(dim // 2, dtype=torch.float)
        i_matrix /= dim / 2
        i_matrix = torch.pow(10000, i_matrix)
        i_matrix = 1 / i_matrix

        # pos matrix
        pos_vec = torch.arange(pos_len, dtype=torch.float)
        out = pos_vec.unsqueeze(1) @ i_matrix.unsqueeze(0)

        # odd/even pos embedding
        position_emb[:, 0::2] = torch.sin(out)
        position_emb[:, 1::2] = torch.cos(out)

        return position_emb

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        pos_enc = self.positional_encodings[:seq_len, :]

        # Add positional encodings to the input embeddings
        x = x + pos_enc

        # Apply mask if provided
        if mask is not None:
            # Reshape mask and multiply
            mask = mask.unsqueeze(-1).expand_as(x)
            x = x * mask

        return self.dropout(x)