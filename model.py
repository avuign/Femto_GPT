import math

import torch
import torch.nn as nn


# Model
class Femto_Chatbot(nn.Module):
    def __init__(self, voc_size, context_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.token_embedding = torch.nn.Embedding(voc_size, embedding_dim)
        self.pos_embedding = torch.nn.Embedding(context_size, embedding_dim)

        self.input_embedding = self.token_embedding + self.pos_embedding

        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

        self.P = nn.Linear(embedding_dim, voc_size)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.pos_E = nn.Embedding(context_size, embedding_dim)

    def forward(self, tokens):
        seq_len = tokens.shape[-1]
        x = self.E(tokens)

        positions = torch.arange(seq_len)
        x = x + self.pos_E(positions)

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.embedding_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        x = weights @ V
        x = self.ff(x)
        logits = self.P(x)
        return logits
