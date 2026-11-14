import torch
import torch.nn as nn
import numpy as np

# 1. TCN Block
class ChausalConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        # Pad on left for causality
        x_pad = nn.functional.pad(x, (self.pad, 0))
        out = self.act(self.norm(self.conv(x_pad)))
        return out + res

class TrajPilotTCN(nn.Module):
    def __init__(self, in_features=16, hidden_dim=64, num_layers=4, future_steps=30, target_dim=2):
        super().__init__()
        layers = []
        c_in = in_features
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(ChausalConv1dBlock(c_in, hidden_dim, kernel_size=3, dilation=dilation))
            c_in = hidden_dim
        self.net = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, future_steps * target_dim)
        )
        self.future_steps = future_steps
        self.target_dim = target_dim

    def forward(self, x):
        # x: (B, 30, 16) -> permute to (B, 16, 30)
        x_t = x.permute(0, 2, 1)
        feat_map = self.net(x_t) # (B, hidden, 30)
        context = feat_map[:, :, -1] # last temporal state
        out = self.decoder(context)
        return out.view(x.shape[0], self.future_steps, self.target_dim)

# 2. Transformer Encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TrajPilotTransformer(nn.Module):
    def __init__(self, in_features=16, d_model=64, nhead=4, num_layers=2, future_steps=30, target_dim=2):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.pos = PositionalEncoding(d_model, max_len=30)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, future_steps * target_dim)
        )
        self.future_steps = future_steps
        self.target_dim = target_dim

    def forward(self, x):
        x_emb = self.pos(self.proj(x))
        encoded = self.transformer(x_emb) # (B, 30, d_model)
        context = encoded[:, -1, :] # last token representation
        out = self.decoder(context)
        return out.view(x.shape[0], self.future_steps, self.target_dim)

# Test forward passes
dummy = torch.randn(8, 30, 16)
tcn = TrajPilotTCN()
trans = TrajPilotTransformer()
out_tcn = tcn(dummy)
out_trans = trans(dummy)
print("TCN params:", sum(p.numel() for p in tcn.parameters()), "out shape:", out_tcn.shape)
print("Transformer params:", sum(p.numel() for p in trans.parameters()), "out shape:", out_trans.shape)
print("Forward passes successful!")
