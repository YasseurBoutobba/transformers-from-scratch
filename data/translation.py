from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import tqdm
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset

ds: DatasetDict = load_dataset("bentrevett/multi30k")

SRC_LANG = "en"
TGT_LANG = "de"
SRC_VOCAB_SIZE = 8000
TGT_VOCAB_SIZE = 16000
MAX_LEN = 128


def build_tokenizer(lang, vocab_size):
    model_path = f"data/multi30k_{lang}_tokenizer.model"
    if not Path(model_path).exists():
        print(f"Building {lang} tokenizer...")
        Path("data").mkdir(exist_ok=True)
        txt_path = f"data/multi30k_{lang}.txt"
        with open(txt_path, mode="w") as f:
            for row in ds["train"]:
                f.writelines(row[lang].replace("\n", " ") + "\n")
        spm.SentencePieceTrainer.train(
            input=txt_path,
            model_prefix=f"data/multi30k_{lang}_tokenizer",
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
    else:
        print(f"{lang} tokenizer already exists, skipping build.")


def tokenize_and_pad(split, batch_size=1000):
    src_model_path = f"data/multi30k_{SRC_LANG}_tokenizer.model"
    tgt_model_path = f"data/multi30k_{TGT_LANG}_tokenizer.model"
    if not Path(src_model_path).exists():
        build_tokenizer(SRC_LANG, SRC_VOCAB_SIZE)
    if not Path(tgt_model_path).exists():
        build_tokenizer(TGT_LANG, TGT_VOCAB_SIZE)

    sp_src = spm.SentencePieceProcessor(model_file=src_model_path)
    sp_tgt = spm.SentencePieceProcessor(model_file=tgt_model_path)

    src_out_path = f"data/multi30k_{split}_{SRC_LANG}.bin"
    tgt_in_out_path = f"data/multi30k_{split}_{TGT_LANG}_in.bin"
    tgt_out_out_path = f"data/multi30k_{split}_{TGT_LANG}_out.bin"
    lengths_path = f"data/multi30k_{split}_lengths.npy"

    if Path(src_out_path).exists():
        print(f"Tokenized {split} split already exists, skipping.")
        return

    print(f"Tokenizing {split} split...")
    src_texts = ds[split][SRC_LANG]
    tgt_texts = ds[split][TGT_LANG]
    n = len(src_texts)

    src_rows = np.zeros((n, MAX_LEN), dtype=np.uint16)  # pad_id = 0
    tgt_in_rows = np.zeros((n, MAX_LEN), dtype=np.uint16)  # decoder input: <bos> ...
    tgt_out_rows = np.zeros((n, MAX_LEN), dtype=np.uint16)  # decoder target: ... <eos>
    lengths = np.zeros((n, 2), dtype=np.int32)  # (src_len, tgt_len) for masking

    for start in tqdm.tqdm(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        src_batch = [t.replace("\n", " ") for t in src_texts[start:end]]
        tgt_batch = [t.replace("\n", " ") for t in tgt_texts[start:end]]

        src_encoded = sp_src.encode(src_batch, out_type=int)
        tgt_encoded = sp_tgt.encode(tgt_batch, out_type=int)

        for i, (src_ids, tgt_ids) in enumerate(zip(src_encoded, tgt_encoded)):
            row_idx = start + i

            src_full = [2] + src_ids + [3]
            src_full = src_full[:MAX_LEN]
            src_rows[row_idx, : len(src_full)] = src_full
            lengths[row_idx, 0] = len(src_full)

            tgt_in_full = ([2] + tgt_ids)[:MAX_LEN]
            tgt_out_full = (tgt_ids + [3])[:MAX_LEN]
            tgt_in_rows[row_idx, : len(tgt_in_full)] = tgt_in_full
            tgt_out_rows[row_idx, : len(tgt_out_full)] = tgt_out_full
            lengths[row_idx, 1] = len(tgt_in_full)

    src_rows.tofile(src_out_path)
    tgt_in_rows.tofile(tgt_in_out_path)
    tgt_out_rows.tofile(tgt_out_out_path)
    np.save(lengths_path, lengths)
    print(f"Done writing {split} split ({n} examples)")


class TranslationDataset(Dataset):
    def __init__(self, split, max_len=MAX_LEN):
        self.max_len = max_len
        self.src = np.memmap(
            f"data/multi30k_{split}_{SRC_LANG}.bin", dtype=np.uint16, mode="r"
        ).reshape(-1, max_len)
        self.tgt_in = np.memmap(
            f"data/multi30k_{split}_{TGT_LANG}_in.bin", dtype=np.uint16, mode="r"
        ).reshape(-1, max_len)
        self.tgt_out = np.memmap(
            f"data/multi30k_{split}_{TGT_LANG}_out.bin", dtype=np.uint16, mode="r"
        ).reshape(-1, max_len)
        self.lengths = np.load(f"data/multi30k_{split}_lengths.npy")

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        src = torch.from_numpy(self.src[index].astype(np.int64))
        tgt_in = torch.from_numpy(self.tgt_in[index].astype(np.int64))
        tgt_out = torch.from_numpy(self.tgt_out[index].astype(np.int64))
        src_len, tgt_len = self.lengths[index]
        # padding masks: True where real tokens, False where padding
        src_mask = torch.arange(self.max_len) < src_len
        tgt_mask = torch.arange(self.max_len) < tgt_len
        return src, tgt_in, tgt_out, src_mask, tgt_mask


def make_src_mask(src, pad_id=0):
    # src: (batch, seq_len)
    mask = (src != pad_id)               # (batch, seq_len) -- True where real token
    mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    return mask

def make_tgt_mask(tgt_in, pad_id=0):
    seq_len = tgt_in.size(1)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=tgt_in.device)).bool()
    causal = causal.unsqueeze(0).unsqueeze(0)          # (1, 1, seq_len, seq_len)

    pad = (tgt_in != pad_id)                            # (batch, seq_len)
    pad = pad.unsqueeze(1).unsqueeze(2)                  # (batch, 1, 1, seq_len)

    tgt_mask = causal & pad                              # broadcasts to (batch, 1, seq_len, seq_len)
    return tgt_mask