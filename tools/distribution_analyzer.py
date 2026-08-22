#!/usr/bin/env python3
"""Gemma 4 WebGPU Distribution Analyzer v0.1.

Reads either a SafeTensors checkpoint directly or an extracted SafeTensors
header JSON. It never loads tensor payloads into RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def load_header(path: Path):
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8")), None
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RuntimeError("Invalid SafeTensors file: missing 8-byte header length")
        header_len = struct.unpack("<Q", raw)[0]
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise RuntimeError("Invalid SafeTensors file: truncated header")
    return json.loads(raw_header.decode("utf-8")), header_len


def tensor_bytes(info):
    offsets = info.get("data_offsets")
    if offsets and len(offsets) == 2:
        return int(offsets[1]) - int(offsets[0])
    dtype = info.get("dtype")
    shape = info.get("shape", [])
    if dtype not in DTYPE_BYTES:
        raise RuntimeError(f"Cannot estimate bytes for dtype {dtype!r}")
    return math.prod(shape) * DTYPE_BYTES[dtype]


def category(name: str):
    n = name.lower()
    if "audio_tower" in n or ".audio." in n or n.startswith("audio_"):
        return "AUDIO_UNUSED"
    if "vision_tower" in n or ".vision." in n or "multi_modal_projector" in n:
        return "VISION_UNUSED"
    if "embed_tokens_per_layer" in n:
        return "PLE_ON_DEMAND"
    if name.startswith("model.language_model."):
        return "CORE_ALWAYS"
    return "OTHER_REVIEW"


def runtime_usage(name: str, cat: str):
    if cat == "PLE_ON_DEMAND":
        return "token-row fetch; per-layer PLE input"
    if cat in ("AUDIO_UNUSED", "VISION_UNUSED"):
        return "not used by text-only Custom WebGPU runtime"
    if name == "model.language_model.embed_tokens.weight":
        return "token embedding + tied LM head; BF16 in current runtime"
    if name.endswith(".weight_packed"):
        return "GPU-resident packed W4 weight"
    if name.endswith(".weight_scale"):
        return "GPU-resident W4 group scale"
    if name.startswith("model.language_model.layers."):
        return "language-model layer parameter"
    if name.startswith("model.language_model.per_layer_model_projection."):
        return "PLE projection parameter"
    if cat == "CORE_ALWAYS":
        return "language-model core; verify loader usage"
    return "manual review"


def quantization(name: str, info):
    if name.endswith(".weight_packed"):
        return "W4 packed"
    if name.endswith(".weight_scale"):
        return f"W4 scale ({info.get('dtype', '?')})"
    return info.get("dtype", "?")


def distribution(cat: str):
    return {
        "CORE_ALWAYS": "core.safetensors / initial download",
        "PLE_ON_DEMAND": "PLE W4 shards / cacheable on demand",
        "AUDIO_UNUSED": "exclude from text distribution",
        "VISION_UNUSED": "exclude from text distribution",
        "OTHER_REVIEW": "manual review before builder",
    }[cat]


def fmt_bytes(n):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024


def ple_w4_projection(rows, group_size=32, scale_bytes=2):
    """Projected symmetric W4 storage matching current runtime convention.

    4-bit values: 0.5 byte/value.
    One scale per group_size values, default BF16 scale = 2 bytes.
    No zero-point tensor.
    """
    params = sum(math.prod(r["shape"]) for r in rows)
    packed = (params + 1) // 2
    groups = sum(math.ceil(math.prod(r["shape"]) / group_size) for r in rows)
    scales = groups * scale_bytes
    return {"params": params, "packed_bytes": packed, "scale_bytes": scales, "total_bytes": packed + scales}


def main():
    ap = argparse.ArgumentParser(description="Analyze Gemma 4 checkpoint distribution without loading weights")
    ap.add_argument("input", type=Path, help="model.safetensors or extracted header JSON")
    ap.add_argument("--out", type=Path, default=Path("distribution-analysis"), help="output directory")
    ap.add_argument("--ple-group-size", type=int, default=32)
    args = ap.parse_args()

    header, header_len = load_header(args.input)
    rows = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        cat = category(name)
        b = tensor_bytes(info)
        rows.append({
            "tensor_name": name,
            "shape": info.get("shape", []),
            "dtype": info.get("dtype", "?"),
            "stored_bytes": b,
            "category": cat,
            "runtime_usage": runtime_usage(name, cat),
            "quantization": quantization(name, info),
            "distribution": distribution(cat),
            "notes": "",
        })

    totals = defaultdict(int)
    counts = Counter()
    for r in rows:
        totals[r["category"]] += r["stored_bytes"]
        counts[r["category"]] += 1
    total = sum(r["stored_bytes"] for r in rows)
    ple_rows = [r for r in rows if r["category"] == "PLE_ON_DEMAND"]
    ple_proj = ple_w4_projection(ple_rows, args.ple_group_size, 2) if ple_rows else None
    projected_total = totals["CORE_ALWAYS"] + (ple_proj["total_bytes"] if ple_proj else 0)

    report = {
        "format": "gemma4-webgpu-distribution-analysis-v0.1",
        "source": str(args.input),
        "header_length": header_len,
        "tensor_count": len(rows),
        "stored_tensor_bytes": total,
        "categories": {k: {"tensor_count": counts[k], "stored_bytes": totals[k]} for k in [
            "CORE_ALWAYS", "PLE_ON_DEMAND", "VISION_UNUSED", "AUDIO_UNUSED", "OTHER_REVIEW"
        ]},
        "ple_w4_projection": ({**ple_proj, "group_size": args.ple_group_size, "scale_dtype": "BF16", "zero_point": False} if ple_proj else None),
        "projected_text_distribution_bytes": projected_total,
        "assumptions": [
            "Text-only Custom WebGPU distribution excludes AUDIO_UNUSED and VISION_UNUSED.",
            "PLE projection assumes symmetric W4, group size 32 by default, BF16 scale per group, no zero point.",
            "CORE_ALWAYS is classification by current Gemma 4 language-model naming; OTHER_REVIEW must be resolved before building distribution files.",
            "This v0.1 analyzer reads metadata only; it does not validate quantization error or generated-token equivalence.",
        ],
        "tensors": rows,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "distribution-analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.out / "distribution-analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tensor_name", "shape", "dtype", "stored_bytes", "category", "runtime_usage", "quantization", "distribution", "notes"])
        for r in rows:
            w.writerow([r["tensor_name"], json.dumps(r["shape"]), r["dtype"], r["stored_bytes"], r["category"], r["runtime_usage"], r["quantization"], r["distribution"], r["notes"]])

    print("=" * 56)
    print("Gemma 4 WebGPU Distribution Analysis v0.1")
    print("=" * 56)
    print(f"source: {args.input}")
    print(f"tensors: {len(rows)}")
    print(f"stored tensor bytes: {fmt_bytes(total)}")
    print()
    for k in ["CORE_ALWAYS", "PLE_ON_DEMAND", "VISION_UNUSED", "AUDIO_UNUSED", "OTHER_REVIEW"]:
        print(f"{k:18s} {fmt_bytes(totals[k]):>12s}  ({counts[k]} tensors)")
    if ple_proj:
        print("\nPLE projected W4")
        print(f"  packed values : {fmt_bytes(ple_proj['packed_bytes'])}")
        print(f"  BF16 scales   : {fmt_bytes(ple_proj['scale_bytes'])}")
        print(f"  total         : {fmt_bytes(ple_proj['total_bytes'])}")
        if totals["PLE_ON_DEMAND"]:
            print(f"  vs current    : {ple_proj['total_bytes']/totals['PLE_ON_DEMAND']:.3f}x")
    print("\nProjected text distribution (CORE + projected PLE W4)")
    print(f"  {fmt_bytes(projected_total)}")
    if totals["OTHER_REVIEW"]:
        print("\nWARNING: OTHER_REVIEW is non-zero; inspect CSV before building distribution files.")
    print(f"\nJSON: {args.out / 'distribution-analysis.json'}")
    print(f"CSV : {args.out / 'distribution-analysis.csv'}")


if __name__ == "__main__":
    main()
