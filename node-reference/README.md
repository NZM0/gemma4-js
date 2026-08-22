# Node.js TensorFlow.js Reference

This directory is the independent dense/BF16 Gemma 4 reference implementation.

Supported checkpoints:

```text
gemma-4-E2B
gemma-4-E2B-it
```

It intentionally does **not** share implementation files with either browser
runtime.

## What this implementation is for

- readable Gemma 4 architecture reference
- TensorFlow.js debugging / research
- direct SafeTensors loading
- standalone Gemma 4 tokenizer
- no Transformers.js
- no model-header.json

The implementation includes Gemma 4's vertical KV sharing:

```text
layer 13 -> sliding-attention shared K/V
layer 14 -> full-attention shared K/V
layers 15..34 -> reuse the corresponding shared K/V
```

This is part of the model architecture, not merely a generation optimization.

## Run

From the repository root:

```bash
npm install
```

Instruction-tuned checkpoint:

```bash
npm run generate:node -- \
    ./gemma-4-E2B-it/model.safetensors \
    "Explain quantum computing in one sentence." \
    ./gemma-4-E2B-it \
    2 \
    chat
```

Base checkpoint:

```bash
npm run generate:node -- \
    ./gemma-4-E2B/model.safetensors \
    "The future of AI is" \
    ./gemma-4-E2B \
    2 \
    base
```

## Performance

The default backend is the pure-JavaScript TensorFlow.js CPU backend. The
checkpoint is too large to expand all BF16 parameters to float32 and keep them
resident, so weights are streamed one layer at a time. Generation also
recomputes the complete prefix for each output token.

That is deliberately slow but keeps the implementation readable and memory
bounded.

A native `@tensorflow/tfjs-node` backend can be investigated separately as an
optional acceleration path without changing the model equations, but it is not
a default dependency because native binding installation is platform-sensitive,
especially on Apple Silicon.
