# GitHub upload

Create an empty public repository named:

```text
NZM0/gemma4-js
```

Then from this directory:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/NZM0/gemma4-js.git
git push -u origin main
```

Then enable GitHub Pages:

```text
Repository Settings → Pages
Source: Deploy from a branch
Branch: main
Folder: / (root)
```

Expected site:

```text
https://nzm0.github.io/gemma4-js/
```

Large generated model files are intentionally excluded by `.gitignore` and are
hosted at:

```text
https://huggingface.co/NZM0/gemma-4-E2B-it-webgpu-w4
```
