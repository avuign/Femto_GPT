import math

import torch
import torch.nn as nn


# Model
class Femto_Chatbot(nn.Module):
    def __init__(self, voc_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.E = nn.Embedding(voc_size, embedding_dim)

        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

        self.P = nn.Linear(embedding_dim, voc_size)

    def forward(self, tokens):
        seq_len = tokens.shape[-1]
        x = self.E(tokens)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.embedding_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        x = weights @ V
        logits = self.P(x)
        return logits
