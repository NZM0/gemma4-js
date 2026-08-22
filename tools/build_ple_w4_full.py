#!/usr/bin/env python3
"""Gemma 4 PLE W4 Full Builder v0.1.

Streams the full BF16 PLE tensor into cache-friendly W4 shards without loading
4.38 GiB into RAM. The quantization contract is exactly the one already
validated by the PLE W4 fixture/runtime tests:

  signed symmetric W4 [-7, 7], group size 32, BF16 scale, no zero point,
  low nibble first, two's-complement nibble encoding.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, struct, time
from pathlib import Path
import numpy as np

PLE = 'model.language_model.embed_tokens_per_layer.weight'
FORMAT = 'gemma4-ple-w4-sharded-v1'
GROUP = 32
QMAX = 7


def human(n: int) -> str:
    units=['B','KiB','MiB','GiB']
    x=float(n)
    for u in units:
        if x < 1024 or u == units[-1]: return f'{x:.2f} {u}'
        x /= 1024


def read_header(path: Path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        h=json.loads(f.read(n).rstrip(b' ').decode('utf-8'))
    return 8+n,h


def bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits_rne(x):
    u=np.asarray(x,dtype=np.float32).view(np.uint32)
    lsb=(u>>16)&1
    return ((u + np.uint32(0x7fff) + lsb)>>16).astype('<u2')


def quantize_batch(rows_f32):
    b,w=rows_f32.shape
    groups=rows_f32.reshape(b,w//GROUP,GROUP)
    maxabs=np.max(np.abs(groups),axis=2)
    scales=np.where(maxabs==0.0,1.0,maxabs/QMAX).astype(np.float32)
    scale_bits=f32_to_bf16_bits_rne(scales)
    # Quantize using the actually stored BF16 scale, matching runtime behavior.
    stored_scales=bf16_bits_to_f32(scale_bits)
    q=np.clip(np.rint(groups/stored_scales[...,None]),-QMAX,QMAX).astype(np.int8)
    q=q.reshape(b,w)
    n=(q.astype(np.int16)&0x0f).astype(np.uint8)
    packed=(n[:,0::2] | (n[:,1::2]<<4)).copy()
    return packed, scale_bits.copy()


def sha256_file(path: Path, block=8<<20):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b=f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description='Build the full Gemma 4 PLE W4 shard set.')
    ap.add_argument('model',type=Path,help='Original QAT model.safetensors')
    ap.add_argument('-o','--output-dir',type=Path,default=Path('webgpu-distribution/ple-w4'))
    ap.add_argument('--rows-per-shard',type=int,default=4096)
    ap.add_argument('--batch-rows',type=int,default=256)
    ap.add_argument('--verify-samples',type=int,default=64)
    ap.add_argument('--no-hash',action='store_true',help='Skip per-shard SHA-256 (faster, not recommended for final distribution).')
    ap.add_argument('--resume',action='store_true',help='Reuse complete shards of the expected size.')
    args=ap.parse_args()

    data_start,h=read_header(args.model)
    if PLE not in h: raise RuntimeError(f'Tensor not found: {PLE}')
    info=h[PLE]
    shape=list(info['shape'])
    if info['dtype']!='BF16' or len(shape)!=2: raise RuntimeError(f'Unexpected PLE metadata: {info}')
    vocab,width=map(int,shape)
    if width%GROUP: raise RuntimeError(f'width {width} is not divisible by group size {GROUP}')
    if args.rows_per_shard<=0 or args.batch_rows<=0: raise ValueError('row counts must be positive')

    start,end=map(int,info['data_offsets'])
    tensor_offset=data_start+start
    source_bytes=end-start
    expected_source=vocab*width*2
    if source_bytes!=expected_source: raise RuntimeError(f'PLE byte count mismatch: {source_bytes} != {expected_source}')

    packed_row=width//2
    scale_row=(width//GROUP)*2
    row_bytes=packed_row+scale_row
    nshards=math.ceil(vocab/args.rows_per_shard)
    total_out=vocab*row_bytes

    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output_dir/'manifest.json'

    print('='*72)
    print('Gemma 4 PLE W4 Full Builder v0.1')
    print('='*72)
    print('source          :',args.model)
    print('tensor          :',PLE)
    print('shape           :',shape)
    print('source BF16     :',human(source_bytes))
    print('group size      :',GROUP)
    print('quantization    : signed symmetric [-7, 7]')
    print('scale dtype     : BF16')
    print('rows/shard      :',args.rows_per_shard)
    print('shards          :',nshards)
    print('packed/row      :',packed_row,'bytes')
    print('scales/row      :',scale_row,'bytes')
    print('projected total :',human(total_out))
    print()

    mm=np.memmap(args.model,dtype='<u2',mode='r',offset=tensor_offset,shape=(vocab,width))
    shard_entries=[]
    built_rows=0
    t0=time.perf_counter()

    for si in range(nshards):
        r0=si*args.rows_per_shard
        r1=min(vocab,r0+args.rows_per_shard)
        rows=r1-r0
        packed_bytes=rows*packed_row
        scale_bytes=rows*scale_row
        size=packed_bytes+scale_bytes
        name=f'ple-{si:05d}-of-{nshards:05d}.bin'
        path=args.output_dir/name
        part=path.with_suffix(path.suffix+'.part')

        reused=args.resume and path.exists() and path.stat().st_size==size
        if not reused:
            if part.exists(): part.unlink()
            with part.open('w+b') as f:
                f.truncate(size)
                for b0 in range(r0,r1,args.batch_rows):
                    b1=min(r1,b0+args.batch_rows)
                    src=bf16_bits_to_f32(np.asarray(mm[b0:b1],dtype='<u2'))
                    packed,scales=quantize_batch(src)
                    local=b0-r0
                    f.seek(local*packed_row)
                    f.write(packed.tobytes(order='C'))
                    f.seek(packed_bytes + local*scale_row)
                    f.write(scales.astype('<u2',copy=False).tobytes(order='C'))
            os.replace(part,path)

        digest=None if args.no_hash else sha256_file(path)
        built_rows += rows
        elapsed=time.perf_counter()-t0
        rate=built_rows/elapsed if elapsed else 0
        eta=(vocab-built_rows)/rate if rate else 0
        status='reuse' if reused else 'write'
        print(f'[{si+1:3d}/{nshards}] {status:5s} rows {r0:6d}..{r1-1:6d}  {human(size):>10s}  ETA {eta/60:6.1f} min')

        shard_entries.append({
            'index':si,'file':name,'row_start':r0,'row_end':r1,'rows':rows,
            'packed_offset':0,'packed_bytes':packed_bytes,
            'scales_offset':packed_bytes,'scales_bytes':scale_bytes,
            'bytes':size,'sha256':digest,
        })

        # Crash-friendly partial manifest.
        partial={
            'format':FORMAT,'version':1,'complete':False,'tensor':PLE,
            'shape':shape,'source_dtype':'BF16','width':width,'vocab_size':vocab,
            'quantization':{'bits':4,'signed':True,'symmetric':True,'qmin':-7,'qmax':7,
                'group_size':GROUP,'scale_dtype':'BF16','zero_point':False,
                'nibble_order':'low_first','signed_encoding':'two_complement_nibble'},
            'layout':{'kind':'row_sharded_packed_then_scales','rows_per_shard':args.rows_per_shard,
                'packed_row_bytes':packed_row,'scale_row_bytes':scale_row},
            'shards':shard_entries,
        }
        manifest_path.write_text(json.dumps(partial,indent=2)+'\n')

    del mm

    # Deterministic verification samples spread over the whole vocabulary.
    verify=[]
    if args.verify_samples>0:
        sample_ids=np.unique(np.linspace(0,vocab-1,min(args.verify_samples,vocab),dtype=np.int64))
        data_start2,h2=read_header(args.model)
        s2=int(h2[PLE]['data_offsets'][0])
        mm2=np.memmap(args.model,dtype='<u2',mode='r',offset=data_start2+s2,shape=(vocab,width))
        for tid in sample_ids:
            si=int(tid)//args.rows_per_shard
            e=shard_entries[si]
            local=int(tid)-e['row_start']
            p=args.output_dir/e['file']
            src=bf16_bits_to_f32(np.asarray(mm2[int(tid):int(tid)+1],dtype='<u2'))
            ep,es=quantize_batch(src)
            with p.open('rb') as f:
                f.seek(local*packed_row); apacked=f.read(packed_row)
                f.seek(e['scales_offset']+local*scale_row); ascales=f.read(scale_row)
            ok=(apacked==ep.tobytes() and ascales==es.astype('<u2',copy=False).tobytes())
            if not ok: raise RuntimeError(f'Verification failed for token row {tid}')
            verify.append(int(tid))
        del mm2
        print(f'\nverify          : PASS ({len(verify)} distributed rows, byte-identical re-quantization)')

    manifest={
        'format':FORMAT,'version':1,'complete':True,'tensor':PLE,'shape':shape,
        'source_dtype':'BF16','width':width,'vocab_size':vocab,
        'quantization':{'bits':4,'signed':True,'symmetric':True,'qmin':-7,'qmax':7,
            'group_size':GROUP,'scale_dtype':'BF16','zero_point':False,
            'nibble_order':'low_first','signed_encoding':'two_complement_nibble'},
        'layout':{'kind':'row_sharded_packed_then_scales','rows_per_shard':args.rows_per_shard,
            'packed_row_bytes':packed_row,'scale_row_bytes':scale_row,
            'token_to_shard':'floor(token_id / rows_per_shard)'},
        'source_bytes':source_bytes,'packed_bytes':vocab*packed_row,
        'scale_bytes':vocab*scale_row,'total_bytes':total_out,
        'ratio':total_out/source_bytes,'verification_rows':verify,'shards':shard_entries,
    }
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    elapsed=time.perf_counter()-t0
    print('manifest        :',manifest_path)
    print('output total    :',human(total_out))
    print('ratio           :',f'{total_out/source_bytes:.5f}x')
    print('elapsed         :',f'{elapsed/60:.2f} min')
    print('\nPASS: full PLE W4 distribution built successfully.')

if __name__=='__main__': main()
