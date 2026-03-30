# Femto GPT
 
A minimal GPT-2-style language model built from scratch in PyTorch, following Sebastian Raschka's *Build a Large Language Model (From Scratch)*. The goal was to understand every component of the transformer architecture by implementing it myself — no high-level wrappers, no `transformers` library.
 
Trained on plain text (Alice in Wonderland, Shakespeare), it generates new sentences from a prompt.
 
## What's in the box
 
The model is a standard decoder-only transformer:
 
- **Token + positional embeddings** → learned lookup tables
- **Multi-head causal self-attention** → Q/K/V projections, scaled dot-product with causal mask, output projection
- **Feed-forward blocks** → two linear layers with GELU activation
- **Pre-norm residual connections** → LayerNorm before each sub-block
- **Linear output head** → projects back to vocabulary logits
 
Generation supports temperature scaling and top-k sampling.
 
## Departures from the book
 
- **Word-level tokenizer** — the book uses tiktoken (BPE). Here, `data.py` implements a simple regex-based word tokenizer with `<UNK>` and `<ENDOFTEXT>` special tokens.
- **Modular file structure** — code is split into `data.py`, `self_attention.py`, `transformer.py`, `model.py`, `train.py`, and `main.py` rather than a single notebook.
 
## Project structure
 
```
config.py            # Hyperparameters (embedding dim, heads, layers, lr, etc.)
data.py              # Word-level tokenizer + input/target generation
self_attention.py    # Attention primitives + multi-head causal attention block
transformer.py       # GELU, feed-forward layer, LayerNorm, transformer block
model.py             # Femto_GPT model (stacked transformer blocks)
train.py             # Training loop with AdamW
main.py              # Text generation with temperature and top-k sampling
alice.txt            # Training corpus (Alice in Wonderland)
shakespeare.txt      # Training corpus (Shakespeare)
verdict.txt          # Training corpus (Edith Wharton, "The Verdict")
```
 
## Usage
 
**Train:**
```bash
python train.py
```
This trains the model on the corpus specified in `config.py` and saves weights to `femto_gpt.pt`.
 
**Generate:**
```bash
python main.py
```
Loads the trained weights and generates text from a prompt. Edit `main.py` to change the prompt, temperature, and top-k.
 
## Default configuration
 
| Parameter | Value |
|---|---|
| Embedding dim | 768 |
| Hidden dim | 3072 (4×768) |
| Context size | 256 |
| Attention heads | 12 |
| Transformer layers | 12 |
| Dropout | 0.1 |
| Optimizer | AdamW (lr=4e-4, weight_decay=0.1) |
 
## Requirements
 
Python 3.10+, PyTorch.
 
```bash
pip install torch
```
 
## Acknowledgements
 
Architecture and training approach follow Sebastian Raschka's [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) (Manning, 2024). This project is a learning exercise — not an original architecture.
