# Browser TensorFlow.js WebGL — Original Checkpoint

Readable browser reference implementation.

- Backend: TensorFlow.js WebGL
- Model source: original `gemma-4-E2B-it-qat-w4a16-ct/model.safetensors`
- PLE source: original BF16 PLE tensor
- Purpose: research, debugging, architecture inspection
- No Custom WebGPU/WGSL model math

Run from repository root:

```bash
npm run serve
```

Open:

```text
http://localhost:8000/browser-tfjs-webgl-original/
```

Prepare tokenizer files first:

```bash
python tools/prepare_tokenizer.py ./gemma-4-E2B-it-qat-w4a16-ct
```
