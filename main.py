import time

import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from data.tinystories import TokenDataset, build_tokenizer, tokenize_and_flatten
from model.base import casual_mask
from model.config import ModelConfig, device
from model.decoder_only_transformer import DecoderOnlyTransformer


def main():
    build_tokenizer()
    tokenize_and_flatten("train")
    train_ds = TokenDataset("data/tinystories_train.bin", seq_len=256)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

    print(f"Number of training samples: {len(train_ds)}")
    print(f"Number of training batches: {len(train_loader)}")

    config = ModelConfig()

    model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_head=config.n_head,
        n_layer=config.n_layer,
        hidden_dim=config.hidden_dim,
        dropout=0.1,
    ).to(device)

    mask = casual_mask(seq_len=config.seq_len).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    print(f"Training model with {sum(p.numel() for p in model.parameters())} parameters on {device}.")
    model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        for batch_idx, (x, y) in enumerate(progress_bar):
            start = time.time()

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            tqdm.tqdm.write(
                f"Batch {batch_idx}, Loss: {loss.item():.4f}, Time: {time.time() - start:.2f}s"
            )


if __name__ == "__main__":
    main()
