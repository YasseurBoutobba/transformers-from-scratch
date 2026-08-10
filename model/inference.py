import sentencepiece as spm
import torch

from data.translation import SRC_LANG, TGT_LANG, make_src_mask, make_tgt_mask
from model.config import device

src_model_path = f"data/multi30k_{SRC_LANG}_tokenizer.model"
tgt_model_path = f"data/multi30k_{TGT_LANG}_tokenizer.model"
sp_src = spm.SentencePieceProcessor(model_file=src_model_path)
sp_tgt = spm.SentencePieceProcessor(model_file=tgt_model_path)


def load_model_for_inference(path, model_class, config, device):

    checkpoint = torch.load(path, map_location=device)

    model = model_class(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_head=config.n_head,
        n_layer=config.n_layer,
        hidden_dim=config.hidden_dim,
        dropout=0.1,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_translation(
    model,
    prompt: str,
    max_len: int = 128,
):

    model.eval()

    src_ids = [2] + sp_src.encode(prompt, out_type=int) + [3]  # <bos> ... <eos>
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    src_mask = make_src_mask(src).to(device)

    enc_output = model.encoder(src, src_mask)  # (1, src_len, d_model)

    # --- Step 2: decode target tokens one at a time ---
    tgt_ids: list[int] = [
        2
    ]  # start with just <bos>, same as tgt_in's first token during training
    for step in range(max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)  # (1, cur_len)
        tgt_mask = make_tgt_mask(tgt).to(device)

        logits = model.decoder(tgt, enc_output, src_mask, tgt_mask)
        next_logits = logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1).item()

        tgt_ids.append(int(next_token))

        if next_token == 3:
            break

    return sp_tgt.decode(tgt_ids)
