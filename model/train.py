import time
from pathlib import Path

import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from data.tinystories import TokenDataset, build_tokenizer, tokenize_and_flatten
from model.base import casual_mask, load_checkpoint, save_checkpoint
from model.config import ModelConfig, device
from model.transformer import DecoderOnlyTransformer


@torch.no_grad()
def evaluate(model, val_loader, loss_fn, mask, config, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            logits = model(x, mask)
            loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def train():
    build_tokenizer()
    tokenize_and_flatten("train")
    tokenize_and_flatten("validation")
    train_ds = TokenDataset("data/tinystories_train.bin", seq_len=256)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_ds = TokenDataset("data/tinystories_validation.bin", seq_len=256)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    print(f"Number of training samples: {len(train_ds)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation samples: {len(val_ds)}")
    print(f"Number of validation batches: {len(val_loader)}")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"
    best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"

    best_val_loss = float("inf")
    config = ModelConfig()
    if Path(latest_checkpoint_path).exists():
        print(f"Loading checkpoint from {latest_checkpoint_path}...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        config = ModelConfig(**checkpoint["config"])

    model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_head=config.n_head,
        n_layer=config.n_layer,
        hidden_dim=config.hidden_dim,
        dropout=0.1,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    last_epoch, last_step = 0, 0

    if Path(latest_checkpoint_path).exists():
        last_epoch, last_step = load_checkpoint(
            latest_checkpoint_path, model, optimizer, device
        )

    mask = casual_mask(seq_len=config.seq_len).to(device)

    print(
        f"Training model with {sum(p.numel() for p in model.parameters())} parameters on {device}."
    )
    model.train()

    for epoch in range(last_epoch, config.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        for batch_idx, (x, y) in enumerate(progress_bar):
            start = time.time()

            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                logits = model(x, mask)
                loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            progress_bar.set_postfix(loss=loss_value)

            if batch_idx % 200 == 0:
                tqdm.tqdm.write(
                    f"Batch {batch_idx}, Loss: {loss_value:.4f}, Time: {time.time() - start:.2f}s"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    batch_idx,
                    config.__dict__,
                    latest_checkpoint_path,
                )
        epoch_loss /= len(train_loader)
        save_checkpoint(
            model,
            optimizer,
            epoch,
            len(train_loader),
            config.__dict__,
            latest_checkpoint_path,
        )
        tqdm.tqdm.write(f"Epoch {epoch} completed. Average Loss: {epoch_loss:.4f}")
        val_loss = evaluate(model, val_loader, loss_fn, mask, config, device)
        tqdm.tqdm.write(f"Validation Loss after Epoch {epoch}: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                len(train_loader),
                config.__dict__,
                best_checkpoint_path,
            )
            tqdm.tqdm.write(
                f"New best model saved with Validation Loss: {best_val_loss:.4f}"
            )
