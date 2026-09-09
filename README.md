# Transformers from scratch

A from-scratch PyTorch implementation of [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — no Hugging Face `transformers`, no borrowed model code. Every module (attention, positional encoding, encoder/decoder blocks) is written and trained in this repo.

Two trained models share the same building blocks:

| | Model | Task | Dataset | Params | Vocab | Runnable today |
|---|---|---|---|---|---|---|
| 1 | `DecoderOnlyTransformer` | next-token generation | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 7.26M | 8k BPE | train + eval only — no sampling script yet |
| 2 | `EncoderDecoderTransfomer`¹ | EN→DE translation | [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) (29,000 train / 1,014 val examples) | 17.6M | 8k EN / 16k DE BPE | **yes** — `uv run main.py` |

¹ *class name is misspelled in the source (`Transfomer`, missing an `r`) — kept as-is here so it matches what you'd `grep` for.*

Param counts above are `sum(p.numel() for p in model.parameters())` with the default configs in `model/config.py`; both fit comfortably on a laptop GPU/MPS or a free Colab T4.

## Quickstart

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync          # installs torch, datasets, sentencepiece, rich, pydantic, tqdm from uv.lock
uv run main.py   # interactive EN → DE translator (rich TUI)
```

`checkpoints/` and the tokenized `data/*.bin|.model|.vocab` files are **gitignored** — they are not in this repo. A fresh clone has nothing to load, so `uv run main.py` will fail until you either train a model yourself (see [Training](#training)) or place a `checkpoints/best_checkpoint_translation.pt` there yourself.

Once a checkpoint exists:

```
╭──────────────────────────────╮
│ English → German Translator  │
│ Multi30k-trained transformer │
╰──────────────────────────────╯
EN: A man is standing on a bridge
DE: <model output>
EN: exit
```

Decoding is greedy, one token at a time, `max_len=128`, stops on `<eos>`. The model is trained on Multi30k's short, literal image-caption sentences, so it does best on similar phrasing — not open-ended text.

## Architecture

```
tokens ─▶ embedding ─▶ + sinusoidal positional encoding ─▶ dropout
```

**Encoder** (translation model only), `N=4` blocks, each pre-norm residual:
```
x → LayerNorm → self-attention (padding-masked, no causal mask) → +x
  → LayerNorm → feed-forward                                    → +x
```

**Decoder** (both models), `N=4` blocks:
```
x → LayerNorm → self-attention (causal-masked, + padding-masked for translation) → +x
  → LayerNorm → cross-attention (Q=decoder, K/V=encoder output)  [translation only] → +x
  → LayerNorm → feed-forward                                     → +x
```
→ final `LayerNorm` → `Linear(d_model → vocab_size)` → logits.

Defaults (`model/config.py`): `d_model=256, n_head=4 (d_k=64), n_layer=4, hidden_dim=1024, dropout=0.1, lr=3e-4, num_epochs=5`. Translation uses `MAX_LEN=128` padded sequences; generation windows are `seq_len=256`.

Component-level detail:

| Component | File | Notes |
|---|---|---|
| `ScaledDotProductAttention` | `model/attention.py` | `softmax(QKᵀ/√d_k)V`, `masked_fill(mask==0, -inf)` before softmax |
| `MultiHeadAttention` | `model/attention.py` | separate `W_q/W_k/W_v/W_o`, splits into `n_head × d_k`, concatenates back |
| `InputEmbedding` | `model/base.py` | plain `nn.Embedding` — **no** `√d_model` scaling (paper §3.4 has it, this doesn't) |
| `PositionalEncoding` | `model/base.py` | sinusoidal, paper's exact formula, precomputed and registered as a buffer |
| `FeedForward` | `model/base.py` | `Linear → GELU → Linear` (paper uses ReLU) |
| `ResidualConnection` | `model/base.py` | **pre-norm**: `x + dropout(sublayer(LayerNorm(x)))` (paper is post-norm) |
| `casual_mask()` | `model/base.py` | lower-triangular causal mask |
| `EncoderBlock` / `DecoderBlock` / `EncoderDecoderTransfomer`¹ | `model/transformer.py` | translation model; `.encoder()` and `.decoder()` are exposed separately so inference can encode once and decode step-by-step |
| `DecoderOnlyBlock` / `DecoderOnlyTransformer` | `model/transformer.py` | GPT-style generation model |

² *also misspelled in the source (`casual` instead of `causal`) — this is the actual function name, not a typo in this doc.*

## Repository layout

```
.
├── main.py                 # EN → DE translator CLI (rich TUI), loads best_checkpoint_translation.pt
├── model/
│   ├── config.py            # ModelConfig, TranslationConfig (pydantic); auto device select cuda > mps > cpu
│   ├── attention.py          # ScaledDotProductAttention, MultiHeadAttention
│   ├── base.py                 # InputEmbedding, PositionalEncoding, FeedForward, ResidualConnection,
│   │                             # casual_mask(), save_checkpoint()/load_checkpoint()
│   ├── transformer.py          # DecoderOnlyBlock/Transformer, EncoderBlock/DecoderBlock/EncoderDecoderTransfomer
│   ├── train.py                # train_gen(), train_translation(), evaluate_gen(), evaluate_translation()
│   └── inference.py            # load_model_for_inference(), generate_translation() (greedy decode)
├── data/
│   ├── tinystories.py       # download, build 8k BPE tokenizer, flatten to one token stream (.bin), TokenDataset
│   ├── translation.py       # download, build EN(8k)/DE(16k) BPE tokenizers, pad to 128 (.bin), masks
│   ├── multi30k_*.model/.vocab   # tokenizers (gitignored, regenerated by training)
│   ├── multi30k_*.bin/.npy       # tokenized splits (gitignored)
│   └── tinystories_*             # tokenizer + flattened corpus, ~2.7 GB total (gitignored)
├── checkpoints/              # gitignored; created by training
│   ├── best_checkpoint_translation.pt / latest_checkpoint_translation.pt
│   ├── best_checkpoint_gen.pt / latest_checkpoint_gen.pt
│   └── latest_checkpoint.pt   # stale, pre-rename TinyStories checkpoint kept for reference
└── pyproject.toml / uv.lock / .python-version   # Python ≥3.13; torch, datasets, sentencepiece, pydantic, rich, tqdm
```

## Data pipeline

Both tokenizers are SentencePiece BPE with the same fixed special ids: `0=pad, 1=unk, 2=bos, 3=eos`. Tokenizer and `.bin` files are built once and skipped on subsequent runs if already present — delete them under `data/` to force a rebuild.

- **TinyStories** (`data/tinystories.py`): a single 8k-vocab tokenizer over the whole corpus. Every story becomes `[bos] + ids + [eos]`, and all stories are concatenated into one long token stream saved to `data/tinystories_train.bin`. `TokenDataset` slices fixed `seq_len+1` windows out of that stream and shifts them into `(x, y)` pairs — no per-example padding.
- **Multi30k** (`data/translation.py`): two tokenizers, EN (8k) and DE (16k). Each example is stored as `src=[bos]+en+[eos]`, `tgt_in=[bos]+de`, `tgt_out=de+[eos]`, all padded to `MAX_LEN=128`. `make_src_mask` builds a padding mask; `make_tgt_mask` combines the causal mask with a padding mask. Both are boolean `(batch, 1, seq, seq)` tensors.

## Training

There is no training CLI — call the functions directly:

```python
from model.train import train_translation, train_gen

train_translation()   # Multi30k EN→DE — the model main.py uses
train_gen()            # TinyStories LM
```

Both loops: `AdamW`, `CrossEntropyLoss(ignore_index=0)`, gradient clipping at `1.0`, `torch.autocast(dtype=torch.bfloat16)`. Checkpoints save to `checkpoints/` every 200 batches and at each epoch end (`latest_*`), plus whenever validation loss improves (`best_*`); training resumes automatically from `latest_*` if it exists.

**GPU training**: open `train_colab.ipynb` in Colab. It checks for CUDA, mounts Google Drive for checkpoint persistence, clones this repo via a personal access token, runs `uv sync`, then trains — so a disconnect just resumes from the last `latest_checkpoint_*.pt` on Drive.

## Deviations from the paper

| Paper | This repo |
|---|---|
| Post-norm residuals | Pre-norm (`LayerNorm` before the sublayer, not after) |
| ReLU feed-forward | GELU |
| Embeddings scaled by `√d_model` | No scaling |
| Adam, warmup + decay LR schedule, label smoothing | `AdamW`, fixed `lr=3e-4`, no warmup, no label smoothing |
| Beam search decoding | Greedy only (`argmax` each step) |
| Base model: 6 layers, `d_model=512`, 8 heads | 4 layers, `d_model=256`, 4 heads — sized to train on a laptop/T4, not to match paper scale |

## Known rough edges

- `casual_mask` and `EncoderDecoderTransfomer` are genuine typos in the source, not documentation errors — searching for the "correct" spelling will find nothing.
- `train.py` hardcodes `torch.autocast(device_type="mps", ...)` regardless of the auto-selected `device` in `model/config.py`. On CUDA or CPU this needs to be changed by hand (see Troubleshooting).
- The decoder-only (TinyStories) model has no inference/sampling script — only `train_gen()` and `evaluate_gen()` (loss-based) exist. `generate_translation()` only serves the translation model.
- No CLI flags for training — hyperparameters are edited directly in `model/config.py`.

## Troubleshooting

- **`main.py` can't load a model** → `checkpoints/best_checkpoint_translation.pt` doesn't exist. It's gitignored and not shipped in the repo — run `train_translation()` first, or copy in your own checkpoint.
- **First training run is slow** → the dataset download and tokenizer build only happen once; subsequent runs reuse the cached `data/*.bin` / `*.model` files.
- **`autocast` errors outside of MPS** → `device_type="mps"` is hardcoded in `model/train.py` (see [Known rough edges](#known-rough-edges)); change it to `"cuda"` or `"cpu"` to match your actual `device`.
- **Corrupted or stale data** → delete the relevant `data/*.bin`, `data/*.npy`, or `data/*.model` file and re-run; each build step is skipped only when its output already exists.
