#!/usr/bin/env python3
"""
Gemma 4 WebGPU Core Extractor v0.1

Build a text-only CORE SafeTensors file from the original
Gemma 4 E2B-it QAT W4A16 checkpoint.

Design goals:
- Never decode or re-encode tensor values.
- Copy CORE tensor payloads byte-for-byte.
- Build a fresh, valid SafeTensors header with remapped offsets.
- Keep PLE, audio, vision, and duplicate LM-head tensors out of CORE.
- Optionally verify every extracted tensor with SHA-256.

The distribution-analysis.json produced by Distribution Analyzer v0.1
is used as the classification manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Tuple


FORMAT_NAME = "gemma4-webgpu-core"
FORMAT_VERSION = "0.1"

COPY_CHUNK_BYTES = 16 * 1024 * 1024
HASH_CHUNK_BYTES = 16 * 1024 * 1024

# v0.1 Analyzer left these three tensors in OTHER_REVIEW.
# They are intentionally excluded from the text-only WebGPU CORE.
KNOWN_EXCLUDED_OTHER = {
    "lm_head.weight": "duplicate/unused LM head; runtime uses tied model.language_model.embed_tokens.weight",
    "model.embed_audio.embedding_projection.weight": "audio-only projection",
    "model.embed_vision.embedding_projection.weight": "vision-only projection",
}


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def read_safetensors_header(path: Path) -> Tuple[int, int, dict]:
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("File is too small to be a SafeTensors file.")

        header_length = struct.unpack("<Q", prefix)[0]
        raw = f.read(header_length)

        if len(raw) != header_length:
            raise ValueError("SafeTensors header is truncated.")

    # SafeTensors JSON is padded with ASCII spaces.
    header = json.loads(raw.rstrip(b" ").decode("utf-8"))
    data_start = 8 + header_length
    return header_length, data_start, header


def load_analysis(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fmt = data.get("format")
    if fmt != "gemma4-webgpu-distribution-analysis-v0.1":
        raise ValueError(
            "Unsupported analysis format. Expected "
            "'gemma4-webgpu-distribution-analysis-v0.1', "
            f"got {fmt!r}."
        )

    return data


def build_core_selection(analysis: dict) -> Tuple[list[str], dict[str, str]]:
    core = []
    excluded_other = {}
    unknown_other = []

    for row in analysis.get("tensors", []):
        name = row["tensor_name"]
        category = row["category"]

        if category == "CORE_ALWAYS":
            core.append(name)
            continue

        if category == "OTHER_REVIEW":
            if name in KNOWN_EXCLUDED_OTHER:
                excluded_other[name] = KNOWN_EXCLUDED_OTHER[name]
            else:
                unknown_other.append(name)

    if unknown_other:
        message = "\n".join(f"  - {name}" for name in unknown_other)
        raise ValueError(
            "Analysis contains unresolved OTHER_REVIEW tensors.\n"
            "Refusing to build CORE until they are classified:\n"
            f"{message}"
        )

    if not core:
        raise ValueError("No CORE_ALWAYS tensors were found in the analysis.")

    return core, excluded_other


def tensor_payload_span(meta: dict) -> Tuple[int, int]:
    offsets = meta.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(v, int) for v in offsets)
    ):
        raise ValueError(f"Invalid data_offsets: {offsets!r}")

    start, end = offsets
    if start < 0 or end < start:
        raise ValueError(f"Invalid data_offsets: {offsets!r}")

    return start, end


def padded_header_bytes(header: dict) -> bytes:
    # SafeTensors requires an 8-byte-aligned header length.
    raw = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    padding = (-len(raw)) % 8
    if padding:
        raw += b" " * padding

    return raw


def copy_range(
    src: BinaryIO,
    dst: BinaryIO,
    src_absolute_start: int,
    length: int,
    *,
    hasher: hashlib._Hash | None = None,
) -> None:
    src.seek(src_absolute_start)
    remaining = length

    while remaining:
        take = min(COPY_CHUNK_BYTES, remaining)
        block = src.read(take)
        if len(block) != take:
            raise IOError(
                f"Unexpected EOF while copying tensor payload; "
                f"wanted {take} bytes, got {len(block)}."
            )

        dst.write(block)

        if hasher is not None:
            hasher.update(block)

        remaining -= take


def sha256_range(
    f: BinaryIO,
    absolute_start: int,
    length: int,
) -> str:
    h = hashlib.sha256()
    f.seek(absolute_start)
    remaining = length

    while remaining:
        take = min(HASH_CHUNK_BYTES, remaining)
        block = f.read(take)

        if len(block) != take:
            raise IOError(
                f"Unexpected EOF while hashing; wanted {take}, got {len(block)}."
            )

        h.update(block)
        remaining -= take

    return h.hexdigest()


def build_core(
    source_path: Path,
    analysis_path: Path,
    output_path: Path,
    *,
    verify: bool,
) -> dict:
    analysis = load_analysis(analysis_path)
    selected_names, excluded_other = build_core_selection(analysis)

    source_header_length, source_data_start, source_header = (
        read_safetensors_header(source_path)
    )

    source_tensor_names = {
        k for k in source_header.keys()
        if k != "__metadata__"
    }

    missing = [
        name for name in selected_names
        if name not in source_tensor_names
    ]
    if missing:
        preview = "\n".join(f"  - {x}" for x in missing[:20])
        raise ValueError(
            f"{len(missing)} CORE tensors from the analysis are missing "
            f"from the source SafeTensors file.\n{preview}"
        )

    # Preserve analysis order so the generated package is deterministic.
    output_header: Dict[str, dict] = {}
    cursor = 0
    total_payload_bytes = 0

    for name in selected_names:
        meta = source_header[name]
        src_start, src_end = tensor_payload_span(meta)
        length = src_end - src_start

        output_header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [cursor, cursor + length],
        }

        cursor += length
        total_payload_bytes += length

    original_metadata = source_header.get("__metadata__", {})
    safe_original_metadata = {
        str(k): str(v)
        for k, v in original_metadata.items()
    }

    output_header["__metadata__"] = {
        **safe_original_metadata,
        "gemma4_webgpu_format": FORMAT_NAME,
        "gemma4_webgpu_format_version": FORMAT_VERSION,
        "gemma4_webgpu_role": "core",
        "source_checkpoint": source_path.name,
        "tensor_count": str(len(selected_names)),
        "payload_bytes": str(total_payload_bytes),
        "ple_included": "false",
        "audio_included": "false",
        "vision_included": "false",
        "duplicate_lm_head_included": "false",
    }

    header_bytes = padded_header_bytes(output_header)
    output_header_length = len(header_bytes)
    output_data_start = 8 + output_header_length

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Gemma 4 WebGPU Core Extractor v0.1")
    print("=" * 64)
    print(f"source        : {source_path}")
    print(f"analysis      : {analysis_path}")
    print(f"output        : {output_path}")
    print(f"CORE tensors  : {len(selected_names)}")
    print(f"CORE payload  : {human_bytes(total_payload_bytes)}")
    print(f"header        : {human_bytes(output_header_length)}")
    print()

    if excluded_other:
        print("[resolved Analyzer v0.1 OTHER_REVIEW]")
        for name, reason in excluded_other.items():
            print(f"  EXCLUDE {name}")
            print(f"          {reason}")
        print()

    started = time.time()
    source_hashes = {} if verify else None

    with source_path.open("rb") as src, output_path.open("wb") as dst:
        dst.write(struct.pack("<Q", output_header_length))
        dst.write(header_bytes)

        for index, name in enumerate(selected_names, start=1):
            src_meta = source_header[name]
            src_start, src_end = tensor_payload_span(src_meta)
            length = src_end - src_start

            h = hashlib.sha256() if verify else None

            copy_range(
                src,
                dst,
                source_data_start + src_start,
                length,
                hasher=h,
            )

            if verify:
                source_hashes[name] = h.hexdigest()

            if index == 1 or index % 100 == 0 or index == len(selected_names):
                print(
                    f"copy {index:4d}/{len(selected_names)}  "
                    f"{human_bytes(length):>11}  {name}"
                )

    elapsed = time.time() - started
    expected_size = output_data_start + total_payload_bytes
    actual_size = output_path.stat().st_size

    if actual_size != expected_size:
        raise IOError(
            f"Output file size mismatch: expected {expected_size}, "
            f"got {actual_size}."
        )

    print()
    print(f"write complete : {human_bytes(actual_size)}")
    print(f"elapsed        : {elapsed:.2f} s")

    verification = None
    if verify:
        verification = verify_core(
            source_path=source_path,
            core_path=output_path,
            selected_names=selected_names,
            expected_source_hashes=source_hashes,
        )

    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source": str(source_path),
        "analysis": str(analysis_path),
        "core_file": str(output_path),
        "tensor_count": len(selected_names),
        "payload_bytes": total_payload_bytes,
        "file_bytes": actual_size,
        "source_header_length": source_header_length,
        "core_header_length": output_header_length,
        "excluded_other_review": excluded_other,
        "verification": verification,
    }

    manifest_path = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"manifest       : {manifest_path}")

    return manifest


def verify_core(
    *,
    source_path: Path,
    core_path: Path,
    selected_names: Iterable[str],
    expected_source_hashes: dict[str, str] | None = None,
) -> dict:
    src_header_length, src_data_start, src_header = (
        read_safetensors_header(source_path)
    )
    core_header_length, core_data_start, core_header = (
        read_safetensors_header(core_path)
    )

    selected_names = list(selected_names)

    core_names = [
        name for name in core_header.keys()
        if name != "__metadata__"
    ]

    if core_names != selected_names:
        raise AssertionError(
            "CORE tensor name/order mismatch after extraction."
        )

    print()
    print("=" * 64)
    print("VERIFY")
    print("=" * 64)

    with source_path.open("rb") as src, core_path.open("rb") as core:
        for index, name in enumerate(selected_names, start=1):
            src_meta = src_header[name]
            out_meta = core_header[name]

            if src_meta["dtype"] != out_meta["dtype"]:
                raise AssertionError(f"{name}: dtype mismatch.")

            if src_meta["shape"] != out_meta["shape"]:
                raise AssertionError(f"{name}: shape mismatch.")

            src_start, src_end = tensor_payload_span(src_meta)
            out_start, out_end = tensor_payload_span(out_meta)

            src_length = src_end - src_start
            out_length = out_end - out_start

            if src_length != out_length:
                raise AssertionError(f"{name}: payload length mismatch.")

            source_hash = (
                expected_source_hashes[name]
                if expected_source_hashes is not None
                else sha256_range(
                    src,
                    src_data_start + src_start,
                    src_length,
                )
            )

            output_hash = sha256_range(
                core,
                core_data_start + out_start,
                out_length,
            )

            if source_hash != output_hash:
                raise AssertionError(
                    f"{name}: SHA-256 mismatch.\n"
                    f"source: {source_hash}\n"
                    f"core  : {output_hash}"
                )

            if index == 1 or index % 100 == 0 or index == len(selected_names):
                print(f"verify {index:4d}/{len(selected_names)}  PASS  {name}")

    print()
    print("PASS: every CORE tensor is byte-identical to the source checkpoint.")

    return {
        "passed": True,
        "method": "per-tensor SHA-256",
        "tensor_count": len(selected_names),
        "source_header_length": src_header_length,
        "core_header_length": core_header_length,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract CORE_ALWAYS tensors from a Gemma 4 QAT SafeTensors "
            "checkpoint into a new text-only core.safetensors."
        )
    )

    p.add_argument(
        "model",
        type=Path,
        help="Original model.safetensors",
    )
    p.add_argument(
        "analysis",
        type=Path,
        help="distribution-analysis.json from Distribution Analyzer v0.1",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("webgpu-distribution/core.safetensors"),
        help="Output SafeTensors path",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="SHA-256 verify every extracted tensor after writing",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    if not args.analysis.is_file():
        raise FileNotFoundError(args.analysis)

    build_core(
        source_path=args.model,
        analysis_path=args.analysis,
        output_path=args.output,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
