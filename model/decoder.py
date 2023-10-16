import torch
import torch.nn as nn
from torch.nn import functional as F
from GPT.blocks.decoder_layer import Block


class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, dropout, n_layer, n_head):
        super().__init__()

        self.n_embd = n_embd
        self.block_size = block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and token both are (B,T) tensor of intergers
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T,C)
        x = token_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        generated_tokens = []  # Initialize a list to store generated tokens

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # get the predictions
            logits, _ = self.forward(idx_cond)
            # focus only on the last time step - context length = 1
            logits = logits[:, -1, :]  # becomes(B,C)
            # apply softmax to get probabilits
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
