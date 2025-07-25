# GPT-163M-params

A minimal GPT-2 style transformer language model in PyTorch.

## Features

- Configurable transformer (layers, heads, embedding) via [`config.py`](config.py)
- GPT-2 BPE tokenization (`tiktoken`)
- Next-token prediction dataset pipeline
- Simple training, validation, and text generation

## Files

- `config.py`: Hyperparameters
- `dataset.py`: Data loading
- `model.py`: Transformer model
- `train.py`: Training loop
- `utils.py`: Loss, evaluation, generation

## Usage

1. `pip install torch tqdm tiktoken`
2. Put your text in `data.txt`
3. Edit `config.py` as needed
4. Run: `python train.py`

---

For research and educational
