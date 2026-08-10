import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask):
        d_k = key.shape[-1]

        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ value
        return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention_head = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        batch_size, seq_len, _ = q.shape
        _, k_len, _ = k.shape
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        key = key.view(batch_size, k_len, self.n_head, self.d_k).transpose(1, 2)
        value = value.view(batch_size, k_len, self.n_head, self.d_k).transpose(1, 2)

        x, _ = self.attention_head.forward(query, key, value, mask)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.n_head * self.d_k)
        )
        return self.w_o(x)
