import torch
import torch.nn as nn
from GPT.layers.single_head_attention_layer  import Head


class MultiHeadAttention(nn.Module):
    """multiple head of self-attention in paralle"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.heads = nn.ModuleList(
            [Head(self.head_size, self.n_embd, self.block_size, self.dropout) for _ in range(self.num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
