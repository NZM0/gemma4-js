
import {
    GPUOps,
    GPUW4Linear,
    SafeFileReader,
    uploadF32,
    createBuffer,
    readF32,
    topK,
} from "./gpu-runtime.js?v=0.1.1";

const U=GPUBufferUsage;
const HIDDEN=1536, PLE=256, LAYERS=35, HEADS=8, MAX_SEQ_DEFAULT=512;

function finiteStats(a) {
    let finite = 0;
    let nan = 0;
    let inf = 0;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let sumSq = 0;

    for (let i = 0; i < a.length; i++) {
        const v = a[i];

        if (Number.isNaN(v)) {
            nan++;
            continue;
        }

        if (!Number.isFinite(v)) {
            inf++;
            continue;
        }

        finite++;
        min = Math.min(min, v);
        max = Math.max(max, v);
        sum += v;
        sumSq += v * v;
    }

    const mean =
        finite > 0
            ? sum / finite
            : NaN;

    const variance =
        finite > 0
            ? Math.max(
                0,
                sumSq / finite -
                mean * mean
            )
            : NaN;

    return {
        count: a.length,
        finite,
        nan,
        inf,
        min: finite > 0 ? min : NaN,
        max: finite > 0 ? max : NaN,
        mean,
        std:
            Number.isFinite(variance)
                ? Math.sqrt(variance)
                : NaN,
    };
}

function formatStats(s) {
    return (
        `finite=${s.finite}/${s.count} ` +
        `nan=${s.nan} inf=${s.inf} ` +
        `min=${Number(s.min).toExponential(3)} ` +
        `max=${Number(s.max).toExponential(3)} ` +
        `mean=${Number(s.mean).toExponential(3)} ` +
        `std=${Number(s.std).toExponential(3)}`
    );
}

function spec(i) {
    const full=(i+1)%5===0;
    return {
        full,
        headDim:full?512:256,
        intermediate:i<15?6144:12288,
        shared:i>=15,
        storeShared:i===13||i===14,
        attentionType:full?"full":"sliding",
        ropeBase:full?1_000_000:10_000,
        ropeProp:full?0.25:1.0,
    };
}

function linearsForLayer(i) {
    const root=`model.language_model.layers.${i}`, s=spec(i);
    const names={
        q:`${root}.self_attn.q_proj`,
        o:`${root}.self_attn.o_proj`,
        gate:`${root}.mlp.gate_proj`,
        up:`${root}.mlp.up_proj`,
        down:`${root}.mlp.down_proj`,
        pleGate:`${root}.per_layer_input_gate`,
        pleProj:`${root}.per_layer_projection`,
    };
    if(!s.shared) {
        names.k=`${root}.self_attn.k_proj`;
        names.v=`${root}.self_attn.v_proj`;
    }
    return names;
}

async function loadScale(device,reader,name,label=name) {
    const x=await reader.bf16(name);
    return uploadF32(device,x,U.STORAGE,label);
}

async function loadLayer(device,reader,i,progress) {
    const root=`model.language_model.layers.${i}`, s=spec(i);
    const names=linearsForLayer(i), linear={};
    for(const [key,base] of Object.entries(names)) {
        linear[key]=await GPUW4Linear.load(device,reader,base);
        progress?.(`  ${base}`);
    }
    const norm={
        input:await loadScale(device,reader,`${root}.input_layernorm.weight`),
        postAttn:await loadScale(device,reader,`${root}.post_attention_layernorm.weight`),
        preFfw:await loadScale(device,reader,`${root}.pre_feedforward_layernorm.weight`),
        postFfw:await loadScale(device,reader,`${root}.post_feedforward_layernorm.weight`),
        postPle:await loadScale(device,reader,`${root}.post_per_layer_input_norm.weight`),
        q:await loadScale(device,reader,`${root}.self_attn.q_norm.weight`),
        k:s.shared?null:await loadScale(device,reader,`${root}.self_attn.k_norm.weight`),
    };
    const skipScale=await reader.scalarBF16(`${root}.layer_scalar`);
    return {i,s,linear,norm,skipScale};
}

class KVCache {
    constructor(device,headDim,maxSeq,label) {
        this.headDim=headDim;this.maxSeq=maxSeq;this.length=0;
        this.k=createBuffer(device,maxSeq*headDim*4,U.STORAGE|U.COPY_DST|U.COPY_SRC,`${label}:k`);
        this.v=createBuffer(device,maxSeq*headDim*4,U.STORAGE|U.COPY_DST|U.COPY_SRC,`${label}:v`);
    }
    append(encoder,kNew,vNew,rows,startPos) {
        if(startPos!==this.length) throw new Error(`KV append position ${startPos} != cache length ${this.length}`);
        if(startPos+rows>this.maxSeq) throw new Error(`KV cache capacity ${this.maxSeq} exceeded`);
        const off=startPos*this.headDim*4, bytes=rows*this.headDim*4;
        encoder.copyBufferToBuffer(kNew,0,this.k,off,bytes);
        encoder.copyBufferToBuffer(vNew,0,this.v,off,bytes);
        this.length+=rows;
    }
    dispose(){this.k.destroy();this.v.destroy();}
}

export class Gemma4WebGPU {
    constructor(device,reader,{maxSeq=MAX_SEQ_DEFAULT,lmChunkRows=4096,log=()=>{}}={}) {
        Object.assign(this,{device,reader,maxSeq,lmChunkRows,log});
        this.ops=new GPUOps(device);
        this.layers=[];
        this.caches=[];
        this.shared={};
        this.lmChunks=[];
        this.loaded=false;
    }

    async load() {
        this.log("Loading top-level W4A16 projection...");
        this.topProj=await GPUW4Linear.load(
            this.device,this.reader,"model.language_model.per_layer_model_projection"
        );
        this.pleNorm=await loadScale(this.device,this.reader,"model.language_model.per_layer_projection_norm.weight");
        this.finalNorm=await loadScale(this.device,this.reader,"model.language_model.norm.weight");

        this.log("Loading 35 quantized transformer layers...");
        for(let i=0;i<LAYERS;i++) {
            this.log(`Layer ${String(i).padStart(2,"0")} ${spec(i).full?"GLOBAL":"local"}...`);
            const layer=await loadLayer(this.device,this.reader,i);
            this.layers.push(layer);
            if(!layer.s.shared) {
                this.caches[i]=new KVCache(this.device,layer.s.headDim,this.maxSeq,`layer${i}`);
            } else {
                this.caches[i]=null;
            }
        }

        this.log("Loading tied BF16 LM-head embedding in GPU chunks...");
        const info=this.reader.info("model.language_model.embed_tokens.weight");
        const vocab=info.shape[0], width=info.shape[1];
        this.vocab=vocab;
        const rowBytes=width*2;
        for(let start=0;start<vocab;start+=this.lmChunkRows) {
            const rows=Math.min(this.lmChunkRows,vocab-start);
            const ab=await this.reader.rangeInTensor(
                "model.language_model.embed_tokens.weight",start*rowBytes,rows*rowBytes
            );
            const buffer=this.device.createBuffer({
                size:Math.max(4,(ab.byteLength+3)&~3),
                usage:U.STORAGE|U.COPY_DST,
                label:`lm-head:${start}`,
            });
            this.device.queue.writeBuffer(buffer,0,ab);
            this.lmChunks.push({start,rows,buffer});
            if(start%(this.lmChunkRows*8)===0) this.log(`  vocab ${start}..${start+rows-1}`);
        }
        this.loaded=true;
        this.log("GPU model preload complete.");
    }

    resetCache() {
        for(const c of this.caches) if(c)c.length=0;
        this.shared={};
    }

    async inputBuffers(tokenIds) {
        const embedPromise =
            this.reader.bf16Rows(
                "model.language_model.embed_tokens.weight",
                tokenIds
            );

        const pleGpuPromise =
            (typeof this.reader.pleRowsToGpu === "function")
                ? this.reader.pleRowsToGpu(this.device,tokenIds)
                : this.reader
                    .bf16Rows(
                        "model.language_model.embed_tokens_per_layer.weight",
                        tokenIds
                    )
                    .then(
                        ple => uploadF32(
                            this.device,
                            ple,
                            U.STORAGE,
                            "token-ple"
                        )
                    );

        const [embed,ple] =
            await Promise.all([
                embedPromise,
                pleGpuPromise,
            ]);

        return {
            embed:uploadF32(this.device,embed,U.STORAGE,"token-embed"),
            ple,
        };
    }

    encodeLayer(encoder,layer,hidden,perLayer,rows,startPos,garbage,probes=null) {
        const {ops}=this, {s,linear,norm}=layer;
        const nHidden=rows*HIDDEN;
        const trash=(x)=>{ if(x) garbage.push(x); return x; };
        const probe=(name,buffer,count)=>{
            if(probes)probes.push({name,buffer,count});
            return buffer;
        };

        const pre=probe(
            "preAttention",
            ops.rms(encoder,hidden,rows,HIDDEN,norm.input,garbage),
            rows*HIDDEN
        );
        const qRaw=probe(
            "qRaw",
            ops.w4(encoder,linear.q,pre,rows,garbage),
            rows*HEADS*s.headDim
        );
        const q=probe(
            "qNormRoPE",
            ops.normRope(
                encoder,qRaw,rows,HEADS,s.headDim,startPos,norm.q,s.ropeProp,s.ropeBase,garbage
            ),
            rows*HEADS*s.headDim
        );
        trash(qRaw);

        let cache;
        if(s.shared) {
            cache=this.shared[s.attentionType];
            if(!cache) throw new Error(`Missing shared ${s.attentionType} KV cache at layer ${layer.i}`);
        } else {
            const kRaw=probe(
                "kRaw",
                ops.w4(encoder,linear.k,pre,rows,garbage),
                rows*s.headDim
            );
            const vRaw=probe(
                "vRaw",
                ops.w4(encoder,linear.v,pre,rows,garbage),
                rows*s.headDim
            );
            const k=probe(
                "kNormRoPE",
                ops.normRope(
                    encoder,kRaw,rows,1,s.headDim,startPos,norm.k,s.ropeProp,s.ropeBase,garbage
                ),
                rows*s.headDim
            );
            const v=probe(
                "vNorm",
                ops.rms(encoder,vRaw,rows,s.headDim,null,garbage),
                rows*s.headDim
            );
            trash(kRaw);trash(vRaw);
            cache=this.caches[layer.i];
            cache.append(encoder,k,v,rows,startPos);
            trash(k);trash(v);
            if(s.storeShared) this.shared[s.attentionType]=cache;
        }
        trash(pre);

        const context=probe(
            "attentionContext",
            ops.attention(
                encoder,q,cache.k,cache.v,rows,HEADS,s.headDim,startPos+rows,startPos,
                !s.full,512,garbage
            ),
            rows*HEADS*s.headDim
        );
        trash(q);
        const attnProj=probe(
            "attentionOutput",
            ops.w4(encoder,linear.o,context,rows,garbage),
            rows*HIDDEN
        ); trash(context);
        const attnNorm=probe(
            "postAttentionNorm",
            ops.rms(encoder,attnProj,rows,HIDDEN,norm.postAttn,garbage),
            rows*HIDDEN
        );trash(attnProj);
        const attnResidual=probe(
            "attentionResidual",
            ops.addScale(encoder,hidden,attnNorm,nHidden,1,garbage),
            rows*HIDDEN
        );trash(attnNorm);trash(hidden);

        const preFfw=probe(
            "preFfw",
            ops.rms(encoder,attnResidual,rows,HIDDEN,norm.preFfw,garbage),
            rows*HIDDEN
        );
        const gate=probe(
            "mlpGate",
            ops.w4(encoder,linear.gate,preFfw,rows,garbage),
            rows*s.intermediate
        );
        const up=probe(
            "mlpUp",
            ops.w4(encoder,linear.up,preFfw,rows,garbage),
            rows*s.intermediate
        );trash(preFfw);
        const act=probe(
            "mlpActivated",
            ops.geluMul(encoder,gate,up,rows*s.intermediate,garbage),
            rows*s.intermediate
        );trash(gate);trash(up);
        const down=probe(
            "ffwOutput",
            ops.w4(encoder,linear.down,act,rows,garbage),
            rows*HIDDEN
        );trash(act);
        const ffwNorm=probe(
            "postFfwNorm",
            ops.rms(encoder,down,rows,HIDDEN,norm.postFfw,garbage),
            rows*HIDDEN
        );trash(down);
        const ffwResidual=probe(
            "ffwResidual",
            ops.addScale(encoder,attnResidual,ffwNorm,nHidden,1,garbage),
            rows*HIDDEN
        );trash(attnResidual);trash(ffwNorm);

        const pg=probe(
            "pleGate",
            ops.w4(encoder,linear.pleGate,ffwResidual,rows,garbage),
            rows*PLE
        );
        const pm=probe(
            "pleModulated",
            ops.geluPle(encoder,pg,perLayer,rows,layer.i,garbage),
            rows*PLE
        );trash(pg);
        const pp=probe(
            "pleProjected",
            ops.w4(encoder,linear.pleProj,pm,rows,garbage),
            rows*HIDDEN
        );trash(pm);
        const pn=probe(
            "pleNormalized",
            ops.rms(encoder,pp,rows,HIDDEN,norm.postPle,garbage),
            rows*HIDDEN
        );trash(pp);
        const out=probe(
            "output",
            ops.addScale(encoder,ffwResidual,pn,nHidden,layer.skipScale,garbage),
            rows*HIDDEN
        );
        trash(ffwResidual);trash(pn);
        return out;
    }
    async forward(tokenIds,startPos,{diagnostic=false}={}) {
        if(!this.loaded) throw new Error("Model not loaded.");
        const rows=tokenIds.length, inp=await this.inputBuffers(tokenIds);

        // Top-level embedding + PLE projection.
        let encoder=this.device.createCommandEncoder();
        let garbage=[];
        let hidden=this.ops.scale(encoder,inp.embed,rows*HIDDEN,Math.sqrt(HIDDEN),garbage);
        const projected=this.ops.w4(encoder,this.topProj,hidden,rows,garbage);
        const projectedScaled=this.ops.scale(
            encoder,projected,rows*35*PLE,1/Math.sqrt(HIDDEN),garbage
        );
        const perLayer=this.ops.pleMix(
            encoder,projectedScaled,this.pleNorm,inp.ple,rows,garbage
        );
        garbage.push(inp.embed,inp.ple,projected,projectedScaled);
        this.device.queue.submit([encoder.finish()]);
        for(const b of garbage)b.destroy();

        if (diagnostic) {
            const hiddenProbe =
                await readF32(
                    this.device,
                    hidden,
                    rows * HIDDEN
                );

            const pleProbe =
                await readF32(
                    this.device,
                    perLayer,
                    rows * 35 * PLE
                );

            this.log(
                `[diag top hidden] ${formatStats(finiteStats(hiddenProbe))}`
            );

            this.log(
                `[diag top PLE]    ${formatStats(finiteStats(pleProbe))}`
            );
        }

        // One queue submission per layer. Buffers are never read back to CPU.
        for(const layer of this.layers) {
            encoder=this.device.createCommandEncoder();
            garbage=[];

            const stageProbes =
                diagnostic && layer.i === 1
                    ? []
                    : null;

            const next=this.encodeLayer(
                encoder,
                layer,
                hidden,
                perLayer,
                rows,
                startPos,
                garbage,
                stageProbes
            );

            this.device.queue.submit([encoder.finish()]);
            hidden=next;

            if (stageProbes) {
                this.log("[diag layer 01 stages]");

                for (const p of stageProbes) {
                    const values =
                        await readF32(
                            this.device,
                            p.buffer,
                            p.count
                        );

                    const stats =
                        finiteStats(
                            values
                        );

                    this.log(
                        `  ${p.name.padEnd(20)} ${formatStats(stats)}`
                    );

                    if (
                        stats.nan > 0 ||
                        stats.inf > 0
                    ) {
                        for(const b of garbage)b.destroy();

                        throw new Error(
                            `First non-finite Layer 1 stage: ${p.name}.`
                        );
                    }
                }
            }

            for(const b of garbage)b.destroy();

            if (diagnostic) {
                const hiddenValues =
                    await readF32(
                        this.device,
                        hidden,
                        rows * HIDDEN
                    );

                const stats =
                    finiteStats(
                        hiddenValues
                    );

                this.log(
                    `[diag layer ${String(layer.i).padStart(2,"0")}] ` +
                    formatStats(stats)
                );

                if (
                    stats.nan > 0 ||
                    stats.inf > 0
                ) {
                    throw new Error(
                        `First non-finite hidden state detected at layer ${layer.i}.`
                    );
                }
            }
        }

        // perLayer is no longer needed after layer 34 was submitted.
        perLayer.destroy();

        encoder=this.device.createCommandEncoder();
        garbage=[];
        const final=this.ops.rms(encoder,hidden,rows,HIDDEN,this.finalNorm,garbage);
        const last=this.ops.emptyF32(HIDDEN,"last-hidden");
        encoder.copyBufferToBuffer(final,(rows-1)*HIDDEN*4,last,0,HIDDEN*4);
        garbage.push(hidden,final);
        this.device.queue.submit([encoder.finish()]);
        for(const b of garbage)b.destroy();

        return last;
    }
    async logits(lastHidden,{diagnostic=false}={}) {
        const logits=this.ops.emptyF32(this.vocab,"logits");
        const encoder=this.device.createCommandEncoder();
        const garbage=[];
        for(const c of this.lmChunks) {
            this.ops.headChunk(encoder,lastHidden,c.buffer,logits,c.start,c.rows,garbage);
        }
        this.device.queue.submit([encoder.finish()]);
        for(const b of garbage)b.destroy();
        const out=await readF32(this.device,logits,this.vocab);
        logits.destroy();

        if (diagnostic) {
            const stats =
                finiteStats(
                    out
                );

            this.log(
                `[diag logits] ${formatStats(stats)}`
            );

            this.log(
                `[diag logits first 16] ` +
                Array.from(
                    out.slice(
                        0,
                        16
                    )
                )
                .map(
                    x =>
                        Number(x)
                            .toExponential(4)
                )
                .join(" ")
            );
        }

        return out;
    }
    async greedy(
        tokenIds,
        startPos,
        {
            diagnostic = false,
        } = {}
    ) {
        const h =
            await this.forward(
                tokenIds,
                startPos,
                {
                    diagnostic,
                }
            );

        const l =
            await this.logits(
                h,
                {
                    diagnostic,
                }
            );

        h.destroy();

        const finiteLogits =
            l.reduce(
                (
                    n,
                    v
                ) =>
                    n +
                    (
                        Number.isFinite(v)
                            ? 1
                            : 0
                    ),
                0
            );

        if (
            finiteLogits !==
            l.length
        ) {
            throw new Error(
                `Non-finite logits: ${l.length - finiteLogits}/${l.length}`
            );
        }

        const top10 =
            topK(
                l,
                10
            );

        return {
            logits:
                l,
            top10,
            nextTokenId:
                top10[0]
                    .tokenId,
        };
    }

    cacheSummary() {
        let bytes=0, caches=0;
        for(const c of this.caches) if(c) {
            caches++;
            bytes+=c.length*c.headDim*4*2;
        }
        return {caches,bytes};
    }

    dispose() {
        this.topProj?.dispose();
        this.pleNorm?.destroy();this.finalNorm?.destroy();
        for(const l of this.layers) {
            for(const x of Object.values(l.linear)) x.dispose();
            for(const x of Object.values(l.norm)) x?.destroy?.();
        }
        for(const c of this.caches) c?.dispose();
        for(const c of this.lmChunks)c.buffer.destroy();
        this.ops.dispose();
    }
}
