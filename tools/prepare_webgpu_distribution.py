#!/usr/bin/env python3
"""Prepare the complete local Gemma 4 Custom WebGPU distribution.

One command performs the complete offline build:
  1. analyze original QAT checkpoint
  2. extract/verify core.safetensors
  3. quantize + shard the full PLE table
  4. copy standalone tokenizer files

The browser runtime then needs only the generated distribution artifacts.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path


def run(cmd, cwd):
    print('\n' + '='*72)
    print('RUN:', ' '.join(map(str,cmd)))
    print('='*72)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def complete_ple_manifest(path: Path) -> bool:
    try:
        x=json.loads(path.read_text())
        return x.get('format')=='gemma4-ple-w4-sharded-v1' and x.get('complete') is True
    except Exception:
        return False


def main():
    ap=argparse.ArgumentParser(description='Prepare Gemma 4 Custom WebGPU distribution in one command.')
    ap.add_argument('model_dir',type=Path,help='Directory containing model.safetensors and tokenizer files')
    ap.add_argument('--force-core',action='store_true',help='Rebuild core.safetensors even when a previous verified extraction exists')
    ap.add_argument('--no-core-verify',action='store_true',help='Skip per-tensor SHA-256 verification of CORE')
    ap.add_argument('--rows-per-shard',type=int,default=4096)
    ap.add_argument('--batch-rows',type=int,default=256)
    ap.add_argument('--verify-samples',type=int,default=64)
    ap.add_argument('--no-shard-hash',action='store_true')
    args=ap.parse_args()

    root=Path(__file__).resolve().parent.parent
    tools=root/'tools'
    model_dir=args.model_dir.expanduser().resolve()
    model=model_dir/'model.safetensors'
    if not model.is_file():
        raise SystemExit(f'model.safetensors not found: {model}')

    analysis_dir=root/'distribution-analysis'
    analysis=analysis_dir/'distribution-analysis.json'
    dist=root/'webgpu-distribution'
    core=dist/'core.safetensors'
    core_manifest=dist/'core.safetensors.manifest.json'
    ple_dir=dist/'ple-w4'
    ple_manifest=ple_dir/'manifest.json'

    print('='*72)
    print('Gemma 4 Custom WebGPU Distribution Preparation')
    print('='*72)
    print('project :',root)
    print('model   :',model)

    # Analyzer is cheap and gives us the classification manifest used by CORE.
    run([sys.executable, tools/'distribution_analyzer.py', model, '--out', analysis_dir], root)

    if args.force_core or not (core.is_file() and core_manifest.is_file()):
        cmd=[sys.executable,tools/'core_extractor.py',model,analysis,'-o',core]
        if not args.no_core_verify:
            cmd.append('--verify')
        run(cmd,root)
    else:
        print(f'\nCORE reuse: {core}')
        print('Use --force-core to rebuild it.')

    cmd=[
        sys.executable,tools/'build_ple_w4_full.py',model,
        '-o',ple_dir,
        '--rows-per-shard',args.rows_per_shard,
        '--batch-rows',args.batch_rows,
        '--verify-samples',args.verify_samples,
        '--resume',
    ]
    if args.no_shard_hash:
        cmd.append('--no-hash')
    run(cmd,root)

    run([sys.executable,tools/'prepare_tokenizer.py',model_dir],root)

    if not core.is_file(): raise SystemExit('CORE output missing after preparation.')
    if not complete_ple_manifest(ple_manifest): raise SystemExit('PLE manifest is missing/incomplete after preparation.')
    if not (root/'tokenizer/gemma4/tokenizer.json').is_file(): raise SystemExit('Tokenizer output missing after preparation.')

    print('\n'+'='*72)
    print('READY — Gemma 4 Custom WebGPU local distribution')
    print('='*72)
    print('CORE :',core)
    print('PLE  :',ple_dir)
    print('TOK  :',root/'tokenizer/gemma4')
    print('\nNext:')
    print('  npm run serve')
    print('  open http://localhost:8000/custom-webgpu/')
    print('\nIn the UI select:')
    print('  1. webgpu-distribution/core.safetensors')
    print('  2. webgpu-distribution/ple-w4/manifest.json')
    print('  3. webgpu-distribution/ple-w4/  (directory picker)')

if __name__=='__main__':
    main()
