# gemma4-js

> Release candidate: v1.0.0-rc3

A from-scratch JavaScript port / runtime project for **Gemma 4 E2B-it**.

This repository keeps four execution paths intentionally separate so the fast
browser runtime, readable TensorFlow.js references, and the dense Node.js
reference can be studied independently.

## Live site

GitHub Pages:

```text
https://nzm0.github.io/gemma4-js/
```

Public compact model distribution:

```text
https://huggingface.co/NZM0/gemma-4-E2B-it-webgpu-w4
```

Base model:

```text
google/gemma-4-E2B-it-qat-w4a16-ct
```

## Implementations

```text
node-reference/
    TensorFlow.js Node reference
    dense BF16 original checkpoint
    intentionally slow and straightforward

browser-tfjs-webgl-original/
    TensorFlow.js WebGL
    original Google QAT W4A16 checkpoint
    research / debugging reference

browser-tfjs-webgl-distribution/
    TensorFlow.js WebGL
    core.safetensors + W4 PLE shards
    readable reference for the compact distribution format

custom-webgpu/
    Custom WebGPU / WGSL runtime
    core.safetensors + W4 PLE shards
    Local Files OR Remote Hugging Face + Browser Cache
```

The implementations are **not** merged into a common model implementation on
purpose.

## Recommended runtime

For ordinary browser inference use:

```text
custom-webgpu/
```

The published UI defaults to **Remote / Hugging Face** and loads model data
from:

```text
https://huggingface.co/NZM0/gemma-4-E2B-it-webgpu-w4/resolve/main
```

No local model checkpoint is required.

The same runtime retains **Local files** mode for offline use, research,
modified weights, and debugging.

## Compact WebGPU distribution

The original multimodal checkpoint contains roughly 7.7 GiB of tensor payload.
For text-only browser inference this project builds:

```text
CORE                     ~1.73 GiB
quantized PLE W4         ~1.23 GiB
---------------------------------
text-only distribution   ~2.96 GiB
```

The CORE contains the original language-model QAT W4A16 weights required by the
runtime.

The very large BF16 per-layer embedding tensor:

```text
model.language_model.embed_tokens_per_layer.weight
[262144, 8960]
```

is additionally quantized to:

```text
signed symmetric W4
integer range: [-7, 7]
group size: 32
BF16 scale
zero point: none
64 shards
```

The public model files are hosted on Hugging Face rather than GitHub.

## Prepare tokenizer files

For local development from the original checkpoint:

```bash
python tools/prepare_tokenizer.py \
    ./gemma-4-E2B-it-qat-w4a16-ct
```

Prepared tokenizer files are served from:

```text
tokenizer/gemma4/
```

## Build the compact distribution

From the original QAT checkpoint:

```bash
python tools/prepare_webgpu_distribution.py \
    ./gemma-4-E2B-it-qat-w4a16-ct
```

This runs the distribution analysis, CORE extraction/verification, full PLE W4
builder, and tokenizer preparation.

Generated large model files belong in:

```text
webgpu-distribution/
```

and are intentionally ignored by Git.

## Local browser development

Install dependencies for the Node reference:

```bash
npm install
```

Start a static server:

```bash
npm run serve
```

Then open:

```text
http://localhost:8000/
```

The two browser TensorFlow.js pages use a pinned TensorFlow.js 4.22.0 CDN build
so the same static files work on GitHub Pages.

## Custom WebGPU model sources

### Remote / Hugging Face

This is the public default.

```text
https://huggingface.co/NZM0/gemma-4-E2B-it-webgpu-w4/resolve/main
```

The runtime expects:

```text
core.safetensors
ple-w4/manifest.json
ple-w4/ple-xxxxx-of-00064.bin
```

CORE is fetched using HTTP Range requests in 32 MiB chunks. PLE shards are
fetched only as needed. The browser Cache API is used when available.

### Local files

Choose:

```text
webgpu-distribution/core.safetensors
webgpu-distribution/ple-w4/
```

The PLE manifest is detected automatically inside the selected directory.

## Node reference

Example:

```bash
npm run generate:node -- \
    ./gemma-4-E2B-it/model.safetensors \
    "Explain quantum computing in one sentence." \
    ./gemma-4-E2B-it \
    2 \
    chat
```

The Node implementation intentionally uses `@tensorflow/tfjs`, not
`@tensorflow/tfjs-node`, and recomputes the prefix. It is kept as a readable
reference rather than a performance implementation.

## GitHub Pages

This repository is static-site compatible.

Recommended Pages setting:

```text
Settings
→ Pages
→ Deploy from a branch
→ main
→ / (root)
```

`.nojekyll` is included.

## Model license and attribution

The model distribution is derived from:

```text
google/gemma-4-E2B-it-qat-w4a16-ct
```

by Google DeepMind and is distributed separately on Hugging Face under Apache
License 2.0. The Hugging Face model card documents the transformations applied
to the checkpoint.

See:

```text
https://huggingface.co/NZM0/gemma-4-E2B-it-webgpu-w4
```

for the model distribution and its license information.

## Status

Before tagging a final `v1.0.0`, verify:

- Custom WebGPU Remote mode against the public Hugging Face repository.
- A second run uses browser cache as expected.
- GitHub Pages paths work from `https://nzm0.github.io/gemma4-js/`.
- Browser TensorFlow.js WebGL original path still works.
- Browser TensorFlow.js WebGL compact-distribution path still works.
- Custom WebGPU Local mode still works.
