# Mini Transformer From Scratch

This project trains a small decoder-only transformer on the TinyStories dataset.
The entry point is [`main.py`](main.py), which builds the tokenizer, downloads and tokenizes the dataset, and then starts model training.

## Requirements

- Python 3.13 or newer
- [`uv`](https://docs.astral.sh/uv/) installed on your machine

## Setup

From the project root, install the dependencies with:

```bash
uv sync
```

This creates the virtual environment and installs everything listed in `pyproject.toml`.

## Run the Project

Start the full pipeline with:

```bash
uv run main.py
```

When you run this command, the project will:

1. Download the TinyStories dataset.
2. Build the TinyStories tokenizer if it does not already exist.
3. Tokenize and flatten the training split into binary training data.
4. Load the dataset into a PyTorch `DataLoader`.
5. Create the decoder-only transformer model.
6. Start training.

## First Run Notes

- The first run can take longer because the dataset must be downloaded and the tokenizer must be trained.
- Generated files are written under `data/`, including:
	- `data/tinystories_tokenizer.model`
	- `data/tinystories_tokenizer.vocab`
	- `data/tinystories_train.bin`
	- `data/tinystories.txt`
- Later runs reuse the saved tokenizer and tokenized data if those files already exist.

## Project Structure

- [`main.py`](main.py): project entry point
- [`data/tinystories.py`](data/tinystories.py): dataset download, tokenizer build, and tokenization helpers
- [`model/`](model): model configuration and transformer implementation

## Troubleshooting

- If `uv` is not installed, install it first from the official `uv` documentation.
- If you want to reset the data pipeline, delete the generated files in `data/` and run `uv run main.py` again.
