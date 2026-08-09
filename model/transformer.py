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


class EncoderBlock(nn.Module):
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


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = FeedForward(d_model, hidden_dim, dropout)
        self.residual_connection1 = ResidualConnection(d_model, dropout)
        self.residual_connection2 = ResidualConnection(d_model, dropout)
        self.residual_connection3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection1(
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )
        x = self.residual_connection2(
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask),
        )
        x = self.residual_connection3(x, self.feed_forward)

        return x


class EncoderDecoderTransfomer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        n_head,
        hidden_dim,
        n_layer,
        seq_len,
        dropout,
    ):
        super().__init__()
        self.enc_layers = nn.ModuleList(
            [EncoderBlock(d_model, n_head, hidden_dim, dropout) for _ in range(n_layer)]
        )
        self.dec_layers = nn.ModuleList(
            [DecoderBlock(d_model, n_head, hidden_dim, dropout) for _ in range(n_layer)]
        )
        self.input_embedding = InputEmbedding(src_vocab_size, d_model)
        self.src_positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        self.output_embedding = InputEmbedding(tgt_vocab_size, d_model)
        self.tgt_positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_x = self.input_embedding(src)
        enc_x = self.src_positional_encoding(enc_x)

        for layer in self.enc_layers:
            enc_x = layer(src, src_mask)

        encoder_output = enc_x

        dec_x = self.output_embedding(tgt)
        dec_x = self.tgt_positional_encoding(dec_x)

        for layer in self.dec_layers:
            dec_x = layer(dec_x, encoder_output, src_mask, tgt_mask)

        logits = self.output_layer(dec_x)
        return logits
