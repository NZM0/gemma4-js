# Custom WebGPU Runtime

High-performance Gemma 4 runtime using Custom WebGPU/WGSL.

The same runtime supports two model-source modes.

## Local files

Use for research, offline experiments, modified weights, and debugging.

Select:

```text
webgpu-distribution/core.safetensors
webgpu-distribution/ple-w4/   (directory)
```

`manifest.json` is detected automatically inside the PLE directory. This is
intentional: `core.safetensors.manifest.json` is a different file and should
not be selected as the PLE manifest.

## Remote / Hugging Face

Set a base URL such as:

```text
https://huggingface.co/USER/MODEL-REPO/resolve/main
```

The runtime expects:

```text
core.safetensors
ple-w4/manifest.json
ple-w4/ple-xxxxx-of-00064.bin
```

Remote CORE reads use 32 MiB HTTP Range chunks. CORE chunks and complete PLE
shards are stored through the browser Cache API when available. The runtime
falls back to network/memory operation when CacheStorage is unavailable.

The Hugging Face repository must allow browser CORS requests and byte-range
requests for `core.safetensors`.
