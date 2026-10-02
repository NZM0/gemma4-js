
// Step 10.19: GPU-resident Gemma 4 E2B-it QAT W4A16 runtime.
// Heavy model math stays in WebGPU buffers. CPU is used only for
// safetensors byte-range I/O, tiny embedding-row decode, and final Top-K readback.

const U = GPUBufferUsage;

function align4(n) { return Math.max(4, (n + 3) & ~3); }

export function createBuffer(device, size, usage, label="") {
    return device.createBuffer({size: align4(size), usage, label});
}

export function uploadBytes(device, bytes, usage=U.STORAGE, label="") {
    const b=createBuffer(device, bytes.byteLength, usage|U.COPY_DST, label);
    device.queue.writeBuffer(b,0,bytes.buffer,bytes.byteOffset,bytes.byteLength);
    return b;
}

export function uploadF32(device, data, usage=U.STORAGE, label="") {
    return uploadBytes(device,new Uint8Array(data.buffer,data.byteOffset,data.byteLength),usage,label);
}

export function uploadU32(device, data, usage=U.UNIFORM, label="") {
    return uploadBytes(device,new Uint8Array(data.buffer,data.byteOffset,data.byteLength),usage,label);
}

export async function readF32(device, buffer, count) {
    const bytes=count*4;
    const rb=createBuffer(device,bytes,U.COPY_DST|U.MAP_READ,"readback");
    const enc=device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer,0,rb,0,bytes);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out=new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return out;
}

export class SafeFileReader {
    constructor(file) {
        this.file=file;
        this.header=null;
        this.metadata={};
        this.dataStart=null;
    }
    async init() {
        const a=await this.file.slice(0,8).arrayBuffer();
        const headerLength=Number(new DataView(a).getBigUint64(0,true));
        const headerBytes=await this.file.slice(8,8+headerLength).arrayBuffer();
        this.header=JSON.parse(new TextDecoder("utf-8").decode(headerBytes));
        this.metadata=this.header.__metadata__??{};
        this.dataStart=8+headerLength;
        return this;
    }
    info(name) {
        const x=this.header[name];
        if(!x) throw new Error(`Tensor not found: ${name}`);
        return x;
    }
    async bytes(name) {
        const x=this.info(name), [s,e]=x.data_offsets;
        return this.file.slice(this.dataStart+s,this.dataStart+e).arrayBuffer();
    }
    async rangeInTensor(name, byteOffset, byteLength) {
        const x=this.info(name), [s,e]=x.data_offsets;
        if(byteOffset<0 || byteOffset+byteLength>e-s) throw new Error(`Range outside ${name}`);
        return this.file.slice(this.dataStart+s+byteOffset,this.dataStart+s+byteOffset+byteLength).arrayBuffer();
    }
    async bf16(name) {
        return bf16ToF32(await this.bytes(name));
    }
    async scalarBF16(name) {
        return (await this.bf16(name))[0];
    }
    async bf16Rows(name, ids) {
        const x=this.info(name);
        if(x.dtype!=="BF16" || x.shape.length!==2) throw new Error(`${name}: BF16 rank2 expected`);
        const width=x.shape[1], rowBytes=width*2;
        const out=new Float32Array(ids.length*width);
        for(let j=0;j<ids.length;j++) {
            const buf=await this.rangeInTensor(name,ids[j]*rowBytes,rowBytes);
            out.set(bf16ToF32(buf),j*width);
        }
        return out;
    }
}


/**
 * Temporary validation reader for the WebGPU distribution split.
 *
 * CORE tensors come from core.safetensors.
 * The original checkpoint is consulted ONLY for the PLE tensor.
 *
 * This class deliberately presents the same reader interface as
 * SafeFileReader so Gemma4WebGPU itself does not need to know that
 * the model is split across two files.
 */
export class CorePlusOriginalPleReader {
    constructor(coreReader, originalReader) {
        this.coreReader = coreReader;
        this.originalReader = originalReader;
        this.pleName = "model.language_model.embed_tokens_per_layer.weight";

        this.header = {
            ...coreReader.header,
            [this.pleName]: originalReader.info(this.pleName),
        };

        this.metadata = coreReader.metadata;
    }

    static async fromFiles(coreFile, originalFile) {
        const [coreReader, originalReader] = await Promise.all([new SafeFileReader(coreFile).init(), new SafeFileReader(originalFile).init()]);
        const pleName = "model.language_model.embed_tokens_per_layer.weight";

        if (!originalReader.header[pleName]) {
            throw new Error("Original checkpoint does not contain the PLE tensor.");
        }

        if (coreReader.header[pleName]) {
            throw new Error("core.safetensors unexpectedly contains the PLE tensor.");
        }

        const role = coreReader.metadata["gemma4_webgpu_role"];

        if (role != null && role !== "core") {
            throw new Error(
                `Selected CORE file has unexpected role: ${role}`
            );
        }

        return new CorePlusOriginalPleReader(coreReader, originalReader);
    }

    _readerFor(name) {
        return (name === this.pleName) ? this.originalReader : this.coreReader;
    }

    info(name) {
        return this
            ._readerFor(name)
            .info(name);
    }

    async bytes(name) {
        return this
            ._readerFor(name)
            .bytes(name);
    }

    async rangeInTensor(name, byteOffset, byteLength) {
        return this
            ._readerFor(name)
            .rangeInTensor(name, byteOffset, byteLength);
    }

    async bf16(name) {
        return this
            ._readerFor(name)
            .bf16(name);
    }

    async scalarBF16(name) {
        return this
            ._readerFor(name)
            .scalarBF16(name);
    }

    async bf16Rows(name, ids) {
        return this
            ._readerFor(name)
            .bf16Rows(name, ids);
    }

    sourceFor(name) {
        return (name === this.pleName) ? "original checkpoint (PLE only)" : "core.safetensors";
    }
}


/**
 * Small validation-only W4 PLE bundle.
 *
 * This is intentionally separate from SafeFileReader and from the final
 * remote/sharded distribution design. It proves that the real Gemma runtime
 * can consume quantized PLE rows without consulting the original checkpoint.
 */
export class W4PleValidationBundle {
    constructor(manifest, packedFile, scalesFile) {
        this.manifest = manifest;
        this.packedFile = packedFile;
        this.scalesFile = scalesFile;
        this.rowIndex = new Map(manifest.token_ids.map((id, i) => [Number(id), i]));
        this.width = Number(manifest.width);
        this.groupSize = Number(manifest.group_size);
        this.packedRowBytes = Number(manifest.packed_row_bytes);
        this.scaleRowBytes = Number(manifest.scale_row_bytes);
        this.pipelineByDevice = new WeakMap();
    }

    static async fromFiles(manifestFile, packedFile, scalesFile) {
        const manifest = JSON.parse(await manifestFile.text());
        if (manifest.format !== "gemma4-ple-w4-runtime-validation-v0.1") {
            throw new Error(`Unexpected PLE W4 manifest: ${manifest.format}`);
        }
        if (manifest.width !== 8960 || manifest.group_size !== 32) {
            throw new Error("PLE W4 validation bundle has unexpected geometry.");
        }
        return new W4PleValidationBundle(manifest, packedFile, scalesFile);
    }

    _pipeline(device) {
        let pipeline = this.pipelineByDevice.get(device);
        if (pipeline) return pipeline;

        const module = device.createShaderModule({code: /* wgsl */`
struct P {
  rows:u32,
  width:u32,
  group_size:u32,
  groups_per_row:u32,
};
@group(0) @binding(0) var<storage,read> packed:array<u32>;
@group(0) @binding(1) var<storage,read> scales:array<u32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
@group(0) @binding(3) var<uniform> p:P;

fn byte_at(i:u32)->u32 {
  let w=packed[i>>2u];
  return (w>>((i&3u)*8u))&0xffu;
}
fn signed4(n:u32)->i32 {
  return select(i32(n),i32(n)-16,n>=8u);
}
fn scale_at(i:u32)->f32 {
  let w=scales[i>>1u];
  let sh=(i&1u)*16u;
  return bitcast<f32>(((w>>sh)&0xffffu)<<16u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let i=gid.x;
  let total=p.rows*p.width;
  if(i>=total){return;}
  let row=i/p.width;
  let col=i-row*p.width;
  let packed_row_bytes=p.width/2u;
  let byte_index=row*packed_row_bytes+(col>>1u);
  let b=byte_at(byte_index);
  let nib=select(b&0xfu,(b>>4u)&0xfu,(col&1u)==1u);
  let group=row*p.groups_per_row+col/p.group_size;
  out[i]=f32(signed4(nib))*scale_at(group);
}
`});
        pipeline = device.createComputePipeline({
            layout:"auto",
            compute:{module,entryPoint:"main"},
        });
        this.pipelineByDevice.set(device,pipeline);
        return pipeline;
    }

    _dequantRowsToGpu(device, rowCount, packedBytes, scaleBytes) {
        const packedGpu = uploadBytes(device, packedBytes, U.STORAGE, "ple-shard-w4-packed");
        const scalesGpu = uploadBytes(device, scaleBytes, U.STORAGE, "ple-shard-w4-scales");

        const out = createBuffer(
            device,
            rowCount * this.width * 4,
            U.STORAGE | U.COPY_SRC,
            "token-ple-w4-dequant"
        );

        const params = uploadU32(
            device,
            new Uint32Array([rowCount, this.width, this.groupSize, this.width / this.groupSize]),
            U.UNIFORM,
            "ple-shard-w4-params"
        );

        const pipeline = this._pipeline(device);

        const bg = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: packedGpu,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: scalesGpu,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: out,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: params,
                    },
                },
            ],
        });

        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();

        pass.setPipeline(pipeline);

        pass.setBindGroup(0, bg);

        pass.dispatchWorkgroups(Math.ceil(rowCount * this.width / 256));

        pass.end();

        device.queue.submit([enc.finish()]);

        // Submission order guarantees the following Gemma dispatches see the
        // dequantized buffer before these temporary upload buffers disappear.
        packedGpu.destroy();
        scalesGpu.destroy();
        params.destroy();

        return out;
    }


    async rowsToGpu(device, ids) {
        const indices = ids.map(id => {
            const x = this.rowIndex.get(Number(id));
            if (x == null) {
                throw new Error(
                    `PLE W4 validation bundle does not contain token ${id}. ` + `The generated trajectory diverged from the reference, or the bundle needs another row.`
                );
            }
            return x;
        });

        // v0.1 deliberately gathers only requested rows from the small local
        // validation bundle. Final distribution will replace this with shards.
        const packedBytes = new Uint8Array(ids.length*this.packedRowBytes);
        const scaleBytes = new Uint8Array(ids.length*this.scaleRowBytes);

        await Promise.all(indices.map(async (srcIndex,j) => {
            const ps=srcIndex*this.packedRowBytes;
            const ss=srcIndex*this.scaleRowBytes;
            const [pbuf,sbuf]=await Promise.all([
                this.packedFile.slice(ps,ps+this.packedRowBytes).arrayBuffer(),
                this.scalesFile.slice(ss,ss+this.scaleRowBytes).arrayBuffer(),
            ]);
            packedBytes.set(new Uint8Array(pbuf),j*this.packedRowBytes);
            scaleBytes.set(new Uint8Array(sbuf),j*this.scaleRowBytes);
        }));

        const packedGpu=uploadBytes(device,packedBytes,U.STORAGE,"ple-w4-packed");
        const scalesGpu=uploadBytes(device,scaleBytes,U.STORAGE,"ple-w4-scales");
        const out=createBuffer(device, ids.length*this.width*4, U.STORAGE|U.COPY_SRC, "token-ple-w4-dequant");
        const params=uploadU32(
            device,
            new Uint32Array([ids.length, this.width, this.groupSize, this.width/this.groupSize]),
            U.UNIFORM,
            "ple-w4-params"
        );

        const pipeline=this._pipeline(device);
        const bg=device.createBindGroup({
            layout:pipeline.getBindGroupLayout(0),
            entries:[
                {binding:0,resource:{buffer:packedGpu}},
                {binding:1,resource:{buffer:scalesGpu}},
                {binding:2,resource:{buffer:out}},
                {binding:3,resource:{buffer:params}},
            ],
        });
        const enc=device.createCommandEncoder();
        const pass=enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0,bg);
        pass.dispatchWorkgroups(Math.ceil(ids.length*this.width/256));
        pass.end();
        device.queue.submit([enc.finish()]);

        // queue order guarantees later model dispatches see the completed data.
        packedGpu.destroy();
        scalesGpu.destroy();
        params.destroy();
        return out;
    }
}


/**
 * Full local PLE W4 shard reader.
 *
 * Consumes the manifest emitted by tools/build_ple_w4_full.py and a browser
 * FileList containing the 64 shard files. Only rows requested by the current
 * prompt/decode step are copied from the relevant shard(s).
 *
 * This deliberately keeps the already validated WGSL dequantization contract.
 * Remote fetch + persistent browser cache is the next transport layer; the
 * model runtime itself does not need to change for that.
 */
export class W4PleShardBundle extends W4PleValidationBundle {
    constructor(manifest, shardFiles) {
        // Parent owns the validated WGSL pipeline implementation. Its
        // validation-only rowIndex/files are unused by this subclass.
        super(
            {
                token_ids: [],
                width: manifest.width,
                group_size: manifest.quantization.group_size,
                packed_row_bytes: manifest.layout.packed_row_bytes,
                scale_row_bytes: manifest.layout.scale_row_bytes,
            },
            null,
            null
        );

        this.manifest = manifest;
        this.vocabSize = Number(manifest.vocab_size);
        this.rowsPerShard = Number(manifest.layout.rows_per_shard);
        this.shards = manifest.shards;
        this.fileByName = new Map(Array.from(shardFiles).map(file => [file.name, file]));

        const missing = this.shards
            .map(s => s.file)
            .filter(name => !this.fileByName.has(name));

        if (missing.length) {
            throw new Error(
                `PLE shard directory is incomplete: missing ${missing.length} file(s), first: ${missing[0]}`
            );
        }
    }

    static async fromFiles(manifestFile, shardFiles) {
        const manifest = JSON.parse(await manifestFile.text());

        if (manifest.format !== "gemma4-ple-w4-sharded-v1" || manifest.complete !== true) {
            throw new Error(
                `Unexpected/incomplete PLE shard manifest: ${manifest.format}`
            );
        }

        if (Number(manifest.width) !== 8960 || Number(manifest.quantization?.group_size) !== 32 || manifest.layout?.kind !== "row_sharded_packed_then_scales"
        ) {
            throw new Error("PLE shard manifest has unexpected geometry/layout.");
        }

        return new W4PleShardBundle(manifest, shardFiles);
    }

    _location(tokenId) {
        const id = Number(tokenId);

        if (!Number.isInteger(id) || id < 0 || id >= this.vocabSize) {
            throw new Error(
                `PLE token id out of range: ${tokenId}`
            );
        }

        const shardIndex = Math.floor(id / this.rowsPerShard);
        const shard = this.shards[shardIndex];

        if (!shard || id < shard.row_start || id >= shard.row_end) {
            throw new Error(
                `PLE shard manifest cannot resolve token ${id}.`
            );
        }

        return {
            shard,
            localRow: id - Number(shard.row_start),
        };
    }

    async rowsToGpu(device, ids) {
        const packedBytes = new Uint8Array(ids.length * this.packedRowBytes);
        const scaleBytes = new Uint8Array(ids.length * this.scaleRowBytes);

        // Group requests by shard so a prefill touching the same shard does
        // not needlessly create independent lookup bookkeeping.
        const requests = ids.map((id, outputRow) => ({
            id: Number(id),
            outputRow,
                ...this._location(id),
        })
        );

        await Promise.all(
            requests.map(async ({
                outputRow,
                shard,
                localRow,
            }) => {
                const file = this.fileByName.get(shard.file);
                const packedStart = Number(shard.packed_offset) + localRow * this.packedRowBytes;
                const scaleStart = Number(shard.scales_offset) + localRow * this.scaleRowBytes;

                const [pbuf, sbuf] = await Promise.all([
                    file
                        .slice(packedStart, packedStart + this.packedRowBytes)
                        .arrayBuffer(),

                    file
                        .slice(scaleStart, scaleStart + this.scaleRowBytes)
                        .arrayBuffer(),
                ]);

                packedBytes.set(new Uint8Array(pbuf), outputRow * this.packedRowBytes);

                scaleBytes.set(new Uint8Array(sbuf), outputRow * this.scaleRowBytes);
            })
        );

        return this._dequantRowsToGpu(device, ids.length, packedBytes, scaleBytes);
    }

}


const DEFAULT_REMOTE_CHUNK_BYTES = 32 * 1024 * 1024;

class RemoteRangeBlob {
    constructor(
        url,
        {
            chunkBytes = DEFAULT_REMOTE_CHUNK_BYTES,
            cacheName = "gemma4-model-cache-v1",
        } = {}
    ) {
        this.url = url;
        this.chunkBytes = chunkBytes;
        this.cacheName = cacheName;
        this.memory = new Map();
    }

    slice(start, end) {
        return {
            arrayBuffer: () => this.readRange(start, end - start),
        };
    }

    async _cache() {
        if (typeof caches === "undefined") {
            return null;
        }

        return caches.open(this.cacheName);
    }

    _key(index) {
        const encoded = encodeURIComponent(this.url);

        return new URL(
            `./__gemma4_cache__/range?src=${encoded}&chunk=${index}`,
            location.href
        ).href;
    }

    async _chunk(index) {
        if (this.memory.has(index)) {
            return this.memory.get(index);
        }

        const cache = await this._cache();
        const key = this._key(index);

        if (cache) {
            const hit = await cache.match(key);

            if (hit) {
                const buffer = await hit.arrayBuffer();

                this.memory.set(index, buffer);

                return buffer;
            }
        }

        const start = index * this.chunkBytes;
        const end = start + this.chunkBytes - 1;

        const response = await fetch(
            this.url,
            {
                headers: {
                    Range: `bytes=${start}-${end}`,
                },
                cache: "no-store",
            }
        );

        if (response.status !== 206) {
            throw new Error(
                `Remote CORE server did not honor HTTP Range ` + `(status ${response.status}). Expected 206 Partial Content: ${this.url}`
            );
        }

        const buffer = await response.arrayBuffer();

        this.memory.set(index, buffer);

        if (cache) {
            await cache.put(key, new Response(buffer.slice(0)));
        }

        return buffer;
    }

    async readRange(start, length) {
        if (length === 0) {
            return new ArrayBuffer(0);
        }

        const first = Math.floor(start / this.chunkBytes);
        const last = Math.floor((start + length - 1) / this.chunkBytes);

        const chunks = await Promise.all(
            Array.from(
                {
                    length: last - first + 1,
                },
                (_, i) => this._chunk(first + i)
            )
        );

        const out = new Uint8Array(length);

        let written = 0;

        for (let index = first; index <= last; ++index) {
            const chunk = new Uint8Array(chunks[index - first]);
            const chunkStart = index * this.chunkBytes;
            const sourceStart = Math.max(start - chunkStart, 0);
            const sourceEnd = Math.min(start + length - chunkStart, chunk.byteLength);
            const part = chunk.subarray(sourceStart, sourceEnd);

            out.set(part, written);

            written += part.byteLength;
        }

        if (written !== length) {
            throw new Error(
                `Remote range length mismatch: ${written} != ${length}`
            );
        }

        return out.buffer;
    }
}

class RemoteShardStore {
    constructor(
        baseUrl,
        {
            cacheName = "gemma4-model-cache-v1",
        } = {}
    ) {
        this.baseUrl = baseUrl.replace(/\/+$/, "");
        this.cacheName = cacheName;
        this.memory = new Map();
    }

    async _cache() {
        if (typeof caches === "undefined") {
            return null;
        }

        return caches.open(this.cacheName);
    }

    url(name) {
        return `${this.baseUrl}/${name}`;
    }

    async fileBuffer(name) {
        if (this.memory.has(name)) {
            return this.memory.get(name);
        }

        const url = this.url(name);
        const cache = await this._cache();

        if (cache) {
            const hit = await cache.match(url);

            if (hit) {
                const buffer = await hit.arrayBuffer();

                this.memory.set(name, buffer);

                return buffer;
            }
        }

        const response = await fetch(
            url,
            {
                cache: "no-store",
            }
        );

        if (!response.ok) {
            throw new Error(
                `Remote PLE shard fetch failed ${response.status}: ${url}`
            );
        }

        const buffer = await response.arrayBuffer();

        this.memory.set(name, buffer);

        if (cache) {
            await cache.put(url, new Response(buffer.slice(0)));
        }

        return buffer;
    }
}

export class RemoteW4PleShardBundle extends W4PleValidationBundle {
    constructor(manifest, pleBaseUrl) {
        super(
            {
                token_ids: [],
                width: manifest.width,
                group_size: manifest.quantization.group_size,
                packed_row_bytes: manifest.layout.packed_row_bytes,
                scale_row_bytes: manifest.layout.scale_row_bytes,
            },
            null,
            null
        );

        this.manifest = manifest;
        this.vocabSize = Number(manifest.vocab_size);
        this.rowsPerShard = Number(manifest.layout.rows_per_shard);
        this.shards = manifest.shards;
        this.store = new RemoteShardStore(pleBaseUrl);
    }

    static async fromUrl(manifestUrl) {
        const response = await fetch(
            manifestUrl,
            {
                cache: "no-store",
            }
        );

        if (!response.ok) {
            throw new Error(
                `Remote PLE manifest fetch failed ${response.status}: ${manifestUrl}`
            );
        }

        const manifest = await response.json();

        if (manifest.format !== "gemma4-ple-w4-sharded-v1" || manifest.complete !== true) {
            throw new Error(
                `Unexpected/incomplete PLE manifest: ${manifest.format}`
            );
        }

        const base = manifestUrl.replace(/\/[^/]*$/, "");

        return new RemoteW4PleShardBundle(manifest, base);
    }

    _location(tokenId) {
        const id = Number(tokenId);
        const shardIndex = Math.floor(id / this.rowsPerShard);
        const shard = this.shards[shardIndex];

        if (!shard) {
            throw new Error(
                `No PLE shard for token ${id}`
            );
        }

        return {
            shard,
            localRow: id - Number(shard.row_start),
        };
    }

    async rowsToGpu(device, ids) {
        const packedBytes = new Uint8Array(ids.length * this.packedRowBytes);
        const scaleBytes = new Uint8Array(ids.length * this.scaleRowBytes);

        await Promise.all(
            ids.map(async (id, outputRow) => {
                const {
                    shard,
                    localRow,
                } = this._location(id);

                const buffer = await this.store.fileBuffer(shard.file);
                const packedStart = Number(shard.packed_offset) + localRow * this.packedRowBytes;
                const scaleStart = Number(shard.scales_offset) + localRow * this.scaleRowBytes;

                packedBytes.set(
                    new Uint8Array(buffer, packedStart, this.packedRowBytes),
                    outputRow * this.packedRowBytes
                );

                scaleBytes.set(
                    new Uint8Array(buffer, scaleStart, this.scaleRowBytes),
                    outputRow * this.scaleRowBytes
                );
            })
        );

        return this._dequantRowsToGpu(device, ids.length, packedBytes, scaleBytes);
    }
}

export class CorePlusRemoteW4PleReader {
    constructor(coreReader, pleBundle) {
        this.coreReader = coreReader;
        this.pleBundle = pleBundle;
        this.pleName = "model.language_model.embed_tokens_per_layer.weight";

        this.header = {
            ...coreReader.header,
            [this.pleName]: {
                dtype: "W4_SHARDED_REMOTE",
                shape: [pleBundle.vocabSize, pleBundle.width],
                data_offsets: [0, 0],
            },
        };

        this.metadata = coreReader.metadata;
    }

    static async fromBaseUrl(baseUrl) {
        const root = baseUrl.replace(/\/+$/, "");
        const coreUrl = `${root}/core.safetensors`;
        const manifestUrl = `${root}/ple-w4/manifest.json`;

        const [coreReader, pleBundle] = await Promise.all([
            new SafeFileReader(new RemoteRangeBlob(coreUrl)).init(),

            RemoteW4PleShardBundle.fromUrl(manifestUrl),
        ]);

        return new CorePlusRemoteW4PleReader(coreReader, pleBundle);
    }

    info(name) {
        if (name === this.pleName) {
            return this.header[name];
        }

        return this.coreReader.info(name);
    }

    bytes(name) {
        if (name === this.pleName) {
            throw new Error("Raw PLE is unavailable in remote W4 mode.");
        }

        return this.coreReader.bytes(name);
    }

    rangeInTensor(name, offset, length) {
        if (name === this.pleName) {
            throw new Error("Raw PLE is unavailable in remote W4 mode.");
        }

        return this.coreReader.rangeInTensor(name, offset, length);
    }

    bf16(name) {
        if (name === this.pleName) {
            throw new Error("BF16 PLE is unavailable in remote W4 mode.");
        }

        return this.coreReader.bf16(name);
    }

    scalarBF16(name) {
        return this.coreReader.scalarBF16(name);
    }

    bf16Rows(name, ids) {
        if (name === this.pleName) {
            throw new Error("Use pleRowsToGpu() for quantized PLE.");
        }

        return this.coreReader.bf16Rows(name, ids);
    }

    pleRowsToGpu(device, ids) {
        return this.pleBundle.rowsToGpu(device, ids);
    }

    sourceFor(name) {
        return (name === this.pleName)
            ? "remote PLE W4 shards + Browser Cache"
            : "remote core.safetensors range cache";
    }
}

export class CorePlusW4PleShardReader {
    constructor(coreReader, pleBundle) {
        this.coreReader = coreReader;
        this.pleBundle = pleBundle;
        this.pleName = "model.language_model.embed_tokens_per_layer.weight";

        this.header = {
            ...coreReader.header,
        };

        this.header[this.pleName] = {
            dtype: "W4_SHARDED",
            shape: [pleBundle.vocabSize, pleBundle.width],
            data_offsets: [0, 0],
        };

        this.metadata = coreReader.metadata;
    }

    static async fromFiles(coreFile, manifestFile, shardFiles) {
        const [coreReader, pleBundle] = await Promise.all([
            new SafeFileReader(coreFile).init(),

            W4PleShardBundle.fromFiles(manifestFile, shardFiles),
        ]);

        if (coreReader.header["model.language_model.embed_tokens_per_layer.weight"]) {
            throw new Error("core.safetensors unexpectedly contains PLE.");
        }

        return new CorePlusW4PleShardReader(coreReader, pleBundle);
    }

    info(name) {
        if (name === this.pleName) {
            return this.header[name];
        }

        return this.coreReader.info(name);
    }

    bytes(name) {
        if (name === this.pleName) {
            throw new Error("Raw BF16 PLE is unavailable in W4 shard mode.");
        }

        return this.coreReader.bytes(name);
    }

    rangeInTensor(name, offset, length) {
        if (name === this.pleName) {
            throw new Error("Raw BF16 PLE is unavailable in W4 shard mode.");
        }

        return this.coreReader.rangeInTensor(name, offset, length);
    }

    bf16(name) {
        if (name === this.pleName) {
            throw new Error("BF16 PLE is unavailable in W4 shard mode.");
        }

        return this.coreReader.bf16(name);
    }

    scalarBF16(name) {
        return this.coreReader.scalarBF16(name);
    }

    bf16Rows(name, ids) {
        if (name === this.pleName) {
            throw new Error("Use pleRowsToGpu() for quantized PLE.");
        }

        return this.coreReader.bf16Rows(name, ids);
    }

    pleRowsToGpu(device, ids) {
        return this.pleBundle.rowsToGpu(device, ids);
    }

    sourceFor(name) {
        return (name === this.pleName) ? "PLE W4 full shards" : "core.safetensors";
    }

}

export class CorePlusW4PleReader {
    constructor(coreReader, pleBundle) {
        this.coreReader=coreReader;
        this.pleBundle=pleBundle;
        this.pleName="model.language_model.embed_tokens_per_layer.weight";
        this.header={...coreReader.header};
        // Synthetic metadata entry: enough for diagnostics/shape inspection.
        this.header[this.pleName]={
            dtype:"W4_VALIDATION",
            shape:[262144,pleBundle.width],
            data_offsets:[0,0],
        };
        this.metadata=coreReader.metadata;
    }

    static async fromFiles(coreFile, manifestFile, packedFile, scalesFile) {
        const [coreReader,pleBundle]=await Promise.all([
            new SafeFileReader(coreFile).init(),
            W4PleValidationBundle.fromFiles(manifestFile,packedFile,scalesFile),
        ]);
        if(coreReader.header["model.language_model.embed_tokens_per_layer.weight"]) {
            throw new Error("core.safetensors unexpectedly contains PLE.");
        }
        return new CorePlusW4PleReader(coreReader,pleBundle);
    }

    info(name) {
        if(name===this.pleName) return this.header[name];
        return this.coreReader.info(name);
    }
    bytes(name) {
        if(name===this.pleName) throw new Error("Raw BF16 PLE is unavailable in W4 mode.");
        return this.coreReader.bytes(name);
    }
    rangeInTensor(name,o,n) {
        if(name===this.pleName) throw new Error("Raw BF16 PLE is unavailable in W4 mode.");
        return this.coreReader.rangeInTensor(name,o,n);
    }
    bf16(name) {
        if(name===this.pleName) throw new Error("BF16 PLE is unavailable in W4 mode.");
        return this.coreReader.bf16(name);
    }
    scalarBF16(name) { return this.coreReader.scalarBF16(name); }
    bf16Rows(name,ids) {
        if(name===this.pleName) throw new Error("Use pleRowsToGpu() for quantized PLE.");
        return this.coreReader.bf16Rows(name,ids);
    }
    pleRowsToGpu(device,ids) {
        return this.pleBundle.rowsToGpu(device,ids);
    }
    sourceFor(name) {
        return name===this.pleName ? "PLE W4 validation bundle" : "core.safetensors";
    }
}

export function bf16ToF32(buffer) {
    const n=buffer.byteLength/2, out=new Float32Array(n), src=new DataView(buffer);
    const scratch=new ArrayBuffer(4), dv=new DataView(scratch);
    for(let i=0;i<n;i++) {
        dv.setUint32(0,src.getUint16(i*2,true)<<16,true);
        out[i]=dv.getFloat32(0,true);
    }
    return out;
}

const W4_SHADER = /* wgsl */`
struct P {
  m:u32, in_dim:u32, out_dim:u32, packed_in:u32,
  groups:u32, group_size:u32, words_per_group:u32, total:u32,
};
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(2) var<storage,read> s:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:P;
fn scale_at(i:u32)->f32 {
  let word=s[i>>1u]; let sh=(i&1u)*16u;
  return bitcast<f32>(((word>>sh)&0xffffu)<<16u);
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g:vec3<u32>) {
  let flat=g.x; if(flat>=p.total){return;}
  let o=flat%p.out_dim; let r=flat/p.out_dim;
  let xb=r*p.in_dim; let wb=o*p.packed_in; let sb=o*p.groups;
  var acc=0.0;
  for(var gr=0u;gr<p.groups;gr=gr+1u){
    let sc=scale_at(sb+gr);
    let gx=xb+gr*p.group_size;
    let gw=wb+gr*p.words_per_group;
    for(var qword=0u;qword<p.words_per_group;qword=qword+1u){
      let word=w[gw+qword];
      let x0=gx+qword*8u;
      for(var k=0u;k<8u;k=k+1u){
        let q=i32((word>>(k*4u))&15u)-8;
        acc=acc+x[x0+k]*f32(q)*sc;
      }
    }
  }
  y[flat]=acc;
}`;

const RMS_SHADER=/* wgsl */`
struct P { rows:u32, width:u32, has_scale:u32, _pad:u32, eps:f32, _a:f32, _b:f32, _c:f32 };
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> scale:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x; if(i>=p.rows*p.width){return;}
 let r=i/p.width; var ss=0.0;
 for(var k=0u;k<p.width;k=k+1u){let v=x[r*p.width+k]; ss=ss+v*v;}
 let inv=inverseSqrt(ss/f32(p.width)+p.eps);
 let sc=select(1.0,scale[i%p.width],p.has_scale!=0u);
 y[i]=x[i]*inv*sc;
}`;

const ADD_SCALE_SHADER=/* wgsl */`
struct P{ n:u32, _0:u32,_1:u32,_2:u32, scale:f32,_3:f32,_4:f32,_5:f32 };
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i<p.n){y[i]=(a[i]+b[i])*p.scale;}}
`;

const SCALE_SHADER=/* wgsl */`
struct P{ n:u32,_0:u32,_1:u32,_2:u32, scale:f32,_3:f32,_4:f32,_5:f32 };
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read_write> y:array<f32>;
@group(0) @binding(2) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i<p.n){y[i]=a[i]*p.scale;}}
`;

const GELU_MUL_SHADER=/* wgsl */`
struct P{n:u32,_0:u32,_1:u32,_2:u32};
@group(0) @binding(0) var<storage,read> gate:array<f32>;
@group(0) @binding(1) var<storage,read> up:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=p.n){return;} let x=gate[i];
 let c=0.7978845608028654;
 let z=c*(x+0.044715*x*x*x);
 // Some WebGPU backends can produce NaN for tanh() at very large
 // magnitudes even though mathematically tanh(z) is already saturated.
 // Clamping at ±10 changes GELU by a negligible amount while avoiding
 // backend exp overflow / Inf-over-Inf behavior.
 let t=tanh(clamp(z,-10.0,10.0));
 let gelu=0.5*x*(1.0+t);
 y[i]=gelu*up[i];
}`;

const GELU_PLE_SHADER=/* wgsl */`
struct P{rows:u32,layer:u32,width:u32,total:u32};
@group(0) @binding(0) var<storage,read> gate:array<f32>;
@group(0) @binding(1) var<storage,read> ple:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=p.total){return;} let row=i/p.width; let d=i%p.width; let x=gate[i];
 let c=0.7978845608028654;
 let z=c*(x+0.044715*x*x*x);
 let t=tanh(clamp(z,-10.0,10.0));
 let gelu=0.5*x*(1.0+t);
 y[i]=gelu*ple[row*(35u*p.width)+p.layer*p.width+d];
}`;

const NORM_ROPE_SHADER=/* wgsl */`
struct P{
 rows:u32,heads:u32,hd:u32,pos0:u32,
 rope_angles:u32,total:u32,_0:u32,_1:u32,
 eps:f32,base:f32,_2:f32,_3:f32
};
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> scale:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=p.total){return;}
 let per_row=p.heads*p.hd; let row=i/per_row; let rem=i%per_row; let h=rem/p.hd; let d=rem%p.hd;
 let baseidx=row*per_row+h*p.hd; var ss=0.0;
 for(var k=0u;k<p.hd;k=k+1u){let v=x[baseidx+k];ss=ss+v*v;}
 let inv=inverseSqrt(ss/f32(p.hd)+p.eps);
 let half=p.hd/2u; let pair=d%half; let mate=select(d+half,d-half,d>=half);
 let a=x[baseidx+d]*inv*scale[d];
 let b=x[baseidx+mate]*inv*scale[mate];
 var sn=0.0; var cs=1.0;
 if(pair<p.rope_angles){
   let exponent=2.0*f32(pair)/f32(p.hd);
   let theta=f32(p.pos0+row)/pow(p.base,exponent);
   sn=sin(theta);cs=cos(theta);
 }
 if(d<half){y[i]=a*cs-b*sn;} else {y[i]=a*cs+b*sn;}
}`;

const QK_SHADER=/* wgsl */`
struct P{qrows:u32,heads:u32,hd:u32,keylen:u32,qpos0:u32,sliding:u32,window:u32,total:u32};
@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> k:array<f32>;
@group(0) @binding(2) var<storage,read_write> logits:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=p.total){return;}
 let key=i%p.keylen; let t=i/p.keylen; let head=t%p.heads; let qr=t/p.heads;
 let qp=p.qpos0+qr;
 var valid=key<=qp;
 if(p.sliding!=0u){valid=valid && (key+p.window)>qp;}
 if(!valid){logits[i]=-1e30;return;}
 let qb=qr*p.heads*p.hd+head*p.hd; let kb=key*p.hd; var acc=0.0;
 for(var d=0u;d<p.hd;d=d+1u){acc=acc+q[qb+d]*k[kb+d];}
 logits[i]=acc;
}`;

const SOFTMAX_SHADER=/* wgsl */`
struct P{rows:u32,width:u32,_0:u32,_1:u32};
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read_write> y:array<f32>;
@group(0) @binding(2) var<uniform> p:P;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let r=g.x;if(r>=p.rows){return;}let b=r*p.width;var mx=-3.4e38;
 for(var i=0u;i<p.width;i=i+1u){mx=max(mx,x[b+i]);}
 var ss=0.0;for(var i=0u;i<p.width;i=i+1u){ss=ss+exp(x[b+i]-mx);}
 for(var i=0u;i<p.width;i=i+1u){y[b+i]=exp(x[b+i]-mx)/ss;}
}`;

const ATTN_OUT_SHADER=/* wgsl */`
struct P{qrows:u32,heads:u32,hd:u32,keylen:u32,total:u32,_0:u32,_1:u32,_2:u32};
@group(0) @binding(0) var<storage,read> probs:array<f32>;
@group(0) @binding(1) var<storage,read> v:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=p.total){return;}let d=i%p.hd;let t=i/p.hd;let h=t%p.heads;let qr=t/p.heads;
 let pb=(qr*p.heads+h)*p.keylen;var acc=0.0;
 for(var k=0u;k<p.keylen;k=k+1u){acc=acc+probs[pb+k]*v[k*p.hd+d];}
 y[i]=acc;
}`;

const PLE_MIX_SHADER=/* wgsl */`
struct P{n:u32,_0:u32,_1:u32,_2:u32,scale:f32,_3:f32,_4:f32,_5:f32};
@group(0) @binding(0) var<storage,read> proj:array<f32>;
@group(0) @binding(1) var<storage,read> normscale:array<f32>;
@group(0) @binding(2) var<storage,read> tokenple:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:P;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=p.n){return;}let row=i/256u;let d=i%256u;let b=row*256u;var ss=0.0;
 for(var k=0u;k<256u;k=k+1u){let v=proj[b+k];ss=ss+v*v;}
 let n=proj[i]*inverseSqrt(ss/256.0+1e-6)*normscale[d];
 y[i]=(n+tokenple[i]*16.0)*0.7071067811865476;
}`;

const BF16_HEAD_SHADER=/* wgsl */`
struct P{in_dim:u32,rows:u32,out_offset:u32,total:u32,scale:f32,softcap:f32,_0:f32,_1:f32};
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
fn bf(i:u32)->f32{let z=w[i>>1u];let sh=(i&1u)*16u;return bitcast<f32>(((z>>sh)&65535u)<<16u);}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let r=g.x;if(r>=p.rows){return;}var acc=0.0;let b=r*p.in_dim;
 for(var k=0u;k<p.in_dim;k=k+1u){acc=acc+x[k]*bf(b+k);}
 let raw=acc*p.scale;out[p.out_offset+r]=tanh(raw/p.softcap)*p.softcap;
}`;

function pipeline(device, code, label) {
    return device.createComputePipeline({
        label, layout:"auto", compute:{module:device.createShaderModule({code,label}),entryPoint:"main"}
    });
}
function paramBuffer(device, byteLength, fill) {
    const a=new ArrayBuffer(byteLength); fill(new DataView(a));
    return uploadBytes(device,new Uint8Array(a),U.UNIFORM,"params");
}
function bg(device, pipe, entries) {
    return device.createBindGroup({layout:pipe.getBindGroupLayout(0),entries:entries.map((buffer,binding)=>({binding,resource:{buffer}}))});
}
function dispatch(encoder, pipe, bind, n, wg=128) {
    const pass=encoder.beginComputePass();pass.setPipeline(pipe);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(n/wg));pass.end();
}

export class GPUOps {
    constructor(device) {
        this.device=device;
        this.p={
            w4:pipeline(device,W4_SHADER,"w4a16"),
            rms:pipeline(device,RMS_SHADER,"rms"),
            add:pipeline(device,ADD_SCALE_SHADER,"add-scale"),
            scale:pipeline(device,SCALE_SHADER,"scale"),
            gelu:pipeline(device,GELU_MUL_SHADER,"gelu-mul"),
            geluple:pipeline(device,GELU_PLE_SHADER,"gelu-ple"),
            rope:pipeline(device,NORM_ROPE_SHADER,"norm-rope"),
            qk:pipeline(device,QK_SHADER,"qk"),
            softmax:pipeline(device,SOFTMAX_SHADER,"softmax"),
            aout:pipeline(device,ATTN_OUT_SHADER,"attention-out"),
            plemix:pipeline(device,PLE_MIX_SHADER,"ple-mix"),
            head:pipeline(device,BF16_HEAD_SHADER,"bf16-head"),
        };
        this.dummy=uploadF32(device,new Float32Array([1]),U.STORAGE,"dummy-scale");
    }
    track(garbage, buffer) { if (garbage) garbage.push(buffer); return buffer; }
    emptyF32(n,label="") { return createBuffer(this.device,n*4,U.STORAGE|U.COPY_SRC|U.COPY_DST,label); }
    w4(encoder, linear, x, rows, garbage=null) {
        const y=this.emptyF32(rows*linear.outDim,`${linear.name}:out`);
        const p=paramBuffer(this.device,32,d=>{
            const vals=[rows,linear.inDim,linear.outDim,linear.inDim/8,linear.groups,linear.groupSize,linear.groupSize/8,rows*linear.outDim];
            vals.forEach((v,i)=>d.setUint32(i*4,v,true));
        });
        dispatch(encoder,this.p.w4,bg(this.device,this.p.w4,[x,linear.packed,linear.scales,y,p]),rows*linear.outDim);
        this.track(garbage,p); return y;
    }
    rms(encoder,x,rows,width,scale=null,garbage=null) {
        const y=this.emptyF32(rows*width,"rms-out");
        const p=paramBuffer(this.device,32,d=>{
            d.setUint32(0,rows,true);d.setUint32(4,width,true);d.setUint32(8,scale?1:0,true);d.setFloat32(16,1e-6,true);
        });
        dispatch(encoder,this.p.rms,bg(this.device,this.p.rms,[x,scale??this.dummy,y,p]),rows*width);
        this.track(garbage,p);return y;
    }
    addScale(encoder,a,b,n,scale=1,garbage=null) {
        const y=this.emptyF32(n,"add-out");const p=paramBuffer(this.device,32,d=>{d.setUint32(0,n,true);d.setFloat32(16,scale,true);});
        dispatch(encoder,this.p.add,bg(this.device,this.p.add,[a,b,y,p]),n,256);this.track(garbage,p);return y;
    }
    scale(encoder,a,n,s,garbage=null) {
        const y=this.emptyF32(n,"scale-out");const p=paramBuffer(this.device,32,d=>{d.setUint32(0,n,true);d.setFloat32(16,s,true);});
        dispatch(encoder,this.p.scale,bg(this.device,this.p.scale,[a,y,p]),n,256);this.track(garbage,p);return y;
    }
    geluMul(encoder,gate,up,n,garbage=null) {
        const y=this.emptyF32(n,"gelu-mul");const p=uploadU32(this.device,new Uint32Array([n,0,0,0]));
        dispatch(encoder,this.p.gelu,bg(this.device,this.p.gelu,[gate,up,y,p]),n,256);this.track(garbage,p);return y;
    }
    geluPle(encoder,gate,ple,rows,layer,garbage=null) {
        const n=rows*256,y=this.emptyF32(n,"gelu-ple");const p=uploadU32(this.device,new Uint32Array([rows,layer,256,n]));
        dispatch(encoder,this.p.geluple,bg(this.device,this.p.geluple,[gate,ple,y,p]),n,256);this.track(garbage,p);return y;
    }
    normRope(encoder,x,rows,heads,hd,pos0,scale,ropeProp,base,garbage=null) {
        const n=rows*heads*hd,y=this.emptyF32(n,"rope-out");
        const p=paramBuffer(this.device,48,d=>{
            [rows,heads,hd,pos0,Math.floor(ropeProp*hd/2),n,0,0].forEach((v,i)=>d.setUint32(i*4,v,true));
            d.setFloat32(32,1e-6,true);d.setFloat32(36,base,true);
        });
        dispatch(encoder,this.p.rope,bg(this.device,this.p.rope,[x,scale,y,p]),n);this.track(garbage,p);return y;
    }
    attention(encoder,q,kcache,vcache,qrows,heads,hd,keylen,qpos0,sliding,window,garbage=null) {
        const ln=qrows*heads*keylen, logits=this.emptyF32(ln,"attn-logits");
        const p1=uploadU32(this.device,new Uint32Array([qrows,heads,hd,keylen,qpos0,sliding?1:0,window,ln]));
        dispatch(encoder,this.p.qk,bg(this.device,this.p.qk,[q,kcache,logits,p1]),ln);this.track(garbage,p1);
        const probs=this.emptyF32(ln,"attn-probs"),p2=uploadU32(this.device,new Uint32Array([qrows*heads,keylen,0,0]));
        dispatch(encoder,this.p.softmax,bg(this.device,this.p.softmax,[logits,probs,p2]),qrows*heads);this.track(garbage,p2);this.track(garbage,logits);
        const total=qrows*heads*hd,out=this.emptyF32(total,"attn-out"),p3=uploadU32(this.device,new Uint32Array([qrows,heads,hd,keylen,total,0,0,0]));
        dispatch(encoder,this.p.aout,bg(this.device,this.p.aout,[probs,vcache,out,p3]),total);this.track(garbage,p3);this.track(garbage,probs);return out;
    }
    pleMix(encoder,proj,normScale,tokenPle,rows,garbage=null) {
        const n=rows*35*256,y=this.emptyF32(n,"ple-mix");const p=paramBuffer(this.device,32,d=>{d.setUint32(0,n,true);d.setFloat32(16,1,true);});
        dispatch(encoder,this.p.plemix,bg(this.device,this.p.plemix,[proj,normScale,tokenPle,y,p]),n);this.track(garbage,p);return y;
    }
    headChunk(encoder,x,weightChunk,out,outOffset,rows,garbage=null) {
        const p=paramBuffer(this.device,32,d=>{
            d.setUint32(0,1536,true);d.setUint32(4,rows,true);d.setUint32(8,outOffset,true);d.setUint32(12,rows,true);
            d.setFloat32(16,1/Math.sqrt(1536),true);d.setFloat32(20,30,true);
        });
        dispatch(encoder,this.p.head,bg(this.device,this.p.head,[x,weightChunk,out,p]),rows);this.track(garbage,p);
    }
    dispose(){this.dummy.destroy();}
}

export class GPUW4Linear {
    constructor(device,name,inDim,outDim,groupSize,packed,scales) {
        Object.assign(this,{device,name,inDim,outDim,groupSize,packed,scales});
        this.groups=inDim/groupSize;
    }
    static async load(device,reader,base) {
        const pi=reader.info(`${base}.weight_packed`), si=reader.info(`${base}.weight_scale`);
        const outDim=pi.shape[0],inDim=pi.shape[1]*8, groups=si.shape[1],groupSize=inDim/groups;
        const [pb,sb]=await Promise.all([reader.bytes(`${base}.weight_packed`),reader.bytes(`${base}.weight_scale`)]);
        return new GPUW4Linear(device,base,inDim,outDim,groupSize,
            uploadBytes(device,new Uint8Array(pb),U.STORAGE,`${base}:packed`),
            uploadBytes(device,new Uint8Array(sb),U.STORAGE,`${base}:scales`));
    }
    dispose(){this.packed.destroy();this.scales.destroy();}
}

export function topK(logits,k=10) {
    const a=[];

    for(let i=0;i<logits.length;i++) {
        const value=logits[i];

        if(!Number.isFinite(value)) {
            continue;
        }

        if(a.length<k || value>a[a.length-1].value) {
            a.push({
                tokenId:i,
                value
            });

            a.sort((x,y)=> y.value-x.value);

            if(a.length>k) {
                a.pop();
            }
        }
    }

    if(a.length===0) {
        throw new Error("topK(): no finite logits.");
    }

    return a;
}
