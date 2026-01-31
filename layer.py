import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAdaptiveAveragedAttention(nn.Module):
    def __init__(self, in_dim, dim, num_heads):
        super().__init__()
        self.dim = in_dim
        self.num_heads = num_heads

        # 多头线性映射：每个head独立生成 Q/K
        self.q_projs = nn.ModuleList([nn.Linear(in_dim, dim) for _ in range(num_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(in_dim, dim) for _ in range(num_heads)])
        self.v_proj = nn.Linear(in_dim, dim)

        # head 之间的可学习权重 alpha
        self.alpha = nn.Parameter(torch.ones(num_heads))

        # 最后输出的 projection（可选）
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape  # Batch, Nodes, Dim

        A_heads = []
        for h in range(self.num_heads):
            Q = self.q_projs[h](x)  # [B, N, D]
            K = self.k_projs[h](x)  # [B, N, D]
            A_h = torch.bmm(Q, K.transpose(1, 2)) / (D ** 0.5)  # [B, N, N]
            A_heads.append(A_h)

        # Stack and weighted sum
        A_stack = torch.stack(A_heads, dim=1)  # [B, H, N, N]
        alpha = F.softmax(self.alpha, dim=0)   # [H]
        A = torch.einsum('bhnm,h->bnm', A_stack, alpha)  # [B, N, N]

        # 直接作用于原始 x（作为 V）
        V = self.v_proj(x)
        out = torch.bmm(A, V)  # [B, N, D]

        return self.out_proj(out), A


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        # assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = embed_dim

        # projection matrices for Q, K, V
        self.q_proj = nn.Linear(in_dim, self.embed_dim)
        self.k_proj = nn.Linear(in_dim, self.embed_dim)
        self.v_proj = nn.Linear(in_dim, self.embed_dim)

        # output projection
        self.out_proj = nn.Linear(embed_dim*num_heads, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        query, key, value: (batch_size, seq_len, embed_dim)
        mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        """

        B, L, E = x.size()
        H = self.num_heads
        D = self.head_dim

        # 1. Linear projections
        Q = self.q_proj(x)  # (B, L, E)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Reshape to (B, H, L, D)
        Q = Q.view(B, L, H, D).transpose(1, 2)
        K = K.view(B, L, H, D).transpose(1, 2)
        V = V.view(B, L, H, D).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # (B, H, L, L)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        # attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (B, H, L, D)

        # 4. Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(B, L, H*D)
        out = self.out_proj(context)

        return out, torch.mean(attn, dim=1)