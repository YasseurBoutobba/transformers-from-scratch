from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import tqdm
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset

ds: DatasetDict = load_dataset("roneneldan/TinyStories")


def build_tokenizer():
    if not Path("data/tinystories_tokenizer.model").exists():
        print("Building tokenizer...")
        with open("data/tinystories.txt", mode="w") as f:
            for row in ds["train"]:
                f.writelines(row["text"].replace("\n", " ") + "\n")

        spm.SentencePieceTrainer.train(
            input="data/tinystories.txt",
            model_prefix="data/tinystories_tokenizer",
            vocab_size=8000,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
    else:
        print("Tokenizer already exists, skipping build.")


def tokenize_and_flatten(split, batch_size=1000):
    if not Path("data/tinystories_tokenizer.model").exists():
        build_tokenizer()

    sp = spm.SentencePieceProcessor(model_file="data/tinystories_tokenizer.model")

    out_path = "data/tinystories_" + split + ".bin"
    if Path(out_path).exists():
        print(f"Tokenized and flattened {split} split already exists, skipping.")
        return

    print(f"Tokenizing and flattening {split} split...")

    texts = ds[split]["text"]
    n = len(texts)

    with open(out_path, "wb") as f_out:
        for start in tqdm.tqdm(range(0, n, batch_size)):
            batch = texts[start : start + batch_size]
            batch = [t.replace("\n", " ") for t in batch]

            
            encoded_batch = sp.encode(batch, out_type=int)

            chunk_ids = []
            for ids in encoded_batch:
                chunk_ids.append(2)
                chunk_ids.extend(ids)
                chunk_ids.append(3)

            arr = np.array(chunk_ids, dtype=np.uint16)
            arr.tofile(f_out)  # append mode via open 'wb' + repeated writes

    print(f"Done writing {out_path}")


class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, index):
        start = index * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
