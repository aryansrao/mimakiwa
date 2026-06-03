# Mimakiwa AI

A transformer language model built **entirely from scratch** in Rust — no PyTorch, no TensorFlow, no ML frameworks. Ships as a native macOS desktop app.

## What it is

Mimakiwa is a tiny transformer (~3.3M parameters) with a full desktop GUI built on [iced 0.13](https://github.com/iced-rs/iced). On first launch it trains a model from scratch using a built-in seed corpus (~2000 steps), shows live progress with time remaining, and saves the result to `~/.mimakiwa/`. Subsequent launches load the saved model instantly.

No CLI. No Python. Just `cargo run`.

## Architecture

```
Mimakiwa (~3.3M params)
  embed_dim:   256
  n_layers:    4
  n_heads:     4  ──┐
  n_kv_heads:  2  ──┘ GQA: 2× KV-cache compression
  activation:  SwiGLU
  norm:        RMSNorm
  position:    RoPE (no learned positional embeddings)
  vocab:       ~4096 (BPE)
```

### Key components (all from scratch)

- **Tensor engine** (`mimakiwa-core`) — n-dimensional `Vec<f32>` with AVX2+FMA SIMD matmul; NEON auto-vectorization on Apple Silicon via `target-cpu=native`
- **Attention** (`mimakiwa-nn`) — Grouped Query Attention with RoPE and causal masking
- **MLP** — SwiGLU: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`
- **BPE tokenizer** (`mimakiwa-tokenizer`) — trained from scratch on the seed corpus
- **AdamW optimizer** (`mimakiwa-train`) — decoupled weight decay, cosine LR with warmup
- **SEAL self-learning** (`mimakiwa-self-learn`) — learns from every conversation, EWC prevents forgetting
- **Web research** (`mimakiwa-search`) — NLP intent analysis → DuckDuckGo/Bing search → HTML scraping → TF-IDF paragraph extraction → answer synthesis

## Web research pipeline

Mimakiwa classifies query intent (Definition / HowTo / Why / Factual / Comparison / General / Conversational) using real NLP (stopwords, question-word detection, TF-IDF scoring), then fetches and ranks web content to synthesize an answer. Source links are shown inline and are clickable.

## GUI

Built with iced 0.13 on macOS. Obsidian Chrome color theme (Alabaster grey background, Blue Slate accent).

- Chat interface with collapsible sidebar (ChatGPT-style)
- Confidence bars on responses
- Clickable source links — open in browser
- Settings panel: temperature, top-p, max tokens, web research toggle, SEAL toggle
- "Retrain from scratch" button

## Project structure

```
mimakiwa/
├── crates/
│   ├── mimakiwa-core/        # Tensor engine, SIMD matmul, ops
│   ├── mimakiwa-nn/          # GQA attention, SwiGLU MLP, RMSNorm, RoPE
│   ├── mimakiwa-model/       # Full transformer, generation, KV-cache
│   ├── mimakiwa-tokenizer/   # BPE tokenizer from scratch
│   ├── mimakiwa-train/       # AdamW, cosine LR, training loop
│   ├── mimakiwa-self-learn/  # SEAL online learning, EWC, experience replay
│   ├── mimakiwa-search/      # DuckDuckGo/Bing scraping, TF-IDF extraction, knowledge store
│   └── mimakiwa-gui/         # iced 0.13 desktop app (main entry point)
```

## Build & run

Requires Rust 1.75+. `.cargo/config.toml` sets `target-cpu=native` automatically so AVX2+FMA (x86) or NEON (Apple Silicon) is used without any extra flags.

```bash
# Launch the GUI (trains on first run, loads instantly after)
cargo run

# Release build for production
cargo build --release
```

## References

- *Attention Is All You Need* — Vaswani et al. 2017
- *GQA: Training Generalized Multi-Query Transformer Models* — Ainslie et al. 2023
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al. 2021
- *GLU Variants Improve Transformer* (SwiGLU) — Noam Shazeer 2020
- *Decoupled Weight Decay Regularization* (AdamW) — Loshchilov & Hutter 2019
- *Self-Adapting Language Models* (SEAL) — MIT 2025, NeurIPS
- *Overcoming Catastrophic Forgetting in NNs* (EWC) — Kirkpatrick et al. 2017
