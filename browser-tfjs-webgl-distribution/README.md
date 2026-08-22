# Browser TensorFlow.js WebGL — Distribution Format

TensorFlow.js WebGL reference using the same compact model distribution as the
Custom WebGPU runtime.

Model data:

```text
core.safetensors
ple-w4/manifest.json
ple-w4/*.bin
```

PLE W4 rows are dequantized in JavaScript into `Float32Array`, then converted
to TensorFlow.js tensors. This deliberately favors readability over maximum
performance.

Open:

```text
http://localhost:8000/browser-tfjs-webgl-distribution/
```

Select CORE, the PLE manifest, and the PLE shard directory.
