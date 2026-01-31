import torch.nn as nn
from layer import MultiHeadAdaptiveAveragedAttention, MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAdaptiveAveragedAttention(in_dim, out_dim, num_heads)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, int(out_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(out_dim * mlp_ratio), out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, A = self.attn(self.norm1(x))
        # x = x + attn_out
        mlp_out = self.mlp(self.norm2(attn_out))
        x = attn_out + mlp_out
        return x, A

class StructureModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_pooling=True, pool_type='avg'):
        super().__init__()
        self.block = TransformerBlock(in_dim, out_dim, num_heads)
        self.use_pooling = use_pooling

        if use_pooling:
            if pool_type == 'avg':
                # [B, N, D] -> [B, D] by pooling over N
                self.pool = nn.AdaptiveAvgPool1d(1)
            elif pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool1d(1)
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}")

    def forward(self, x):
        x, A = self.block(x)  # x: [B, N, D], A: [B, N, N]
        if self.use_pooling:
            # 转置为 [B, D, N] 以适配 1D pooling，然后 squeeze 回 [B, D]
            x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return x, A