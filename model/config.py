import torch
from pydantic import BaseModel

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class ModelConfig(BaseModel):
    vocab_size: int = 8000
    seq_len: int = 256
    d_model: int = 256
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    hidden_dim: int = 1024
    dropout: float = 0.1
    lr: float = 3e-4
    num_epochs: int = 5


class TranslationConfig(BaseModel):
    src_vocab_size: int = 8000
    tgt_vocab_size: int = 16000
    seq_len: int = 256
    d_model: int = 256
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    hidden_dim: int = 1024
    dropout: float = 0.1
    lr: float = 3e-4
    num_epochs: int = 5
