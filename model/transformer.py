from torch import nn

from model.attention import MultiHeadAttention
from model.base import (
    FeedForward,
    InputEmbedding,
    PositionalEncoding,
    ResidualConnection,
)


class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = FeedForward(d_model, hidden_dim, dropout)
        self.residual_connection1 = ResidualConnection(d_model, dropout)
        self.residual_connection2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask):
        x = self.residual_connection1(x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connection2(x, self.feed_forward)

        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_head, hidden_dim, n_layer, seq_len, dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderOnlyBlock(d_model, n_head, hidden_dim, dropout)
                for _ in range(n_layer)
            ]
        )
        self.input_embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.layer_norm(x)
        logits = self.output_layer(x)
        return logits


