# Transformer-from-Scratch with TinyGrad

A minimal but complete implementation of the original *Attention Is All You Need* Transformer architecture using [TinyGrad](https://github.com/tinygrad/tinygrad).
The repo contains three self-contained examples that demonstrate how to train and use both **decoder-only** and **encoder-decoder** Transformers on very different tasks:

| Example | Task | Model | Dataset | File |
|---|---|---|---|---|
| Adder | Learn to add two 2-digit numbers | Decoder-only | Synthetic arithmetic | `adder.py` |
| Story Generator | Character-level language modelling | Decoder-only | Shakespeare corpus | `story_gen.py` |
| English → Spanish Translator | Machine translation | Encoder-decoder | `okezieowen/english_to_spanish` | `eng_es_translator.py` |

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/andresnowak/transformer-from-scratch-tinygrad.git
cd transformer-from-scratch-tinygrad
uv sync          # or: pip install -e .

# Run any example
python adder.py
python story_gen.py
python eng_es_translator.py
```

---

## Architecture Highlights

* **Transformer** - implementation from [Attention is all you need](https://arxiv.org/pdf/1706.03762)
* **Pure TinyGrad** – Model implemented in Tinygrad completely (but dataset creation uses Numpy, Torch and Hugginface dataset)
* **Pre-norm** residual paths and **scaled-dot-product** attention as in the paper.
* **Encoder-decoder** variant with cross-attention for seq-to-seq tasks.
* **Generation utilities** (`generate`) with top-k sampling and temperature.

---

## File Overview

| File | Purpose |
|---|---|
| `src/transformer.py` | Core model code (`DecoderTransformer`, `EncoderDecoderTransformer`, `MultiHeadAttn`, etc.) |
| `adder.py` | Trains a tiny Transformer to add two 2-digit numbers (e.g., 42 + 17 = 059). |
| `story_gen.py` | Character-level language model trained on Shakespeare; generates text auto-regressively. |
| `eng_es_translator.py` | English → Spanish translation fine-tuned on a small Hugging Face dataset. |
