# Project layout

```text
gemma4-js/
├── .nojekyll
├── .gitignore
├── README.md
├── GITHUB_UPLOAD.md
├── index.html
├── package.json
├── tools/
├── tokenizer/
│   └── gemma4/
├── node-reference/
├── browser-tfjs-webgl-original/
├── browser-tfjs-webgl-distribution/
├── custom-webgpu/
├── distribution-analysis/       # generated/diagnostic
└── webgpu-distribution/         # local generated weights; not committed
```

The four runtime paths are independent by design.
