const tf = globalThis.tf;
import { RMSNorm } from "./rmsnorm.js?v=0.2.2";
import { applyRoPE } from "./rope.js?v=0.2.2";
import { OfficialW4A16Linear } from "./official-quantized-linear.js?v=0.2.2";

function geluTanh(x) {
    return tf.tidy(() => {
        const c = Math.sqrt(2 / Math.PI);
        const x3 = x.pow(tf.scalar(3));
        const inner = x.add(
            x3.mul(0.044715)
        ).mul(c);

        return x.mul(0.5).mul(
            tf.tanh(inner.clipByValue(-10, 10)).add(1)
        );
    });
}

function repeatKVHeads(x, targetHeads) {
    const [batch, seq, kvHeads, headDim] = x.shape;
    const repeats = targetHeads / kvHeads;
    if (!Number.isInteger(repeats)) {
        throw new Error("Invalid GQA head ratio.");
    }
    if (repeats === 1) return x.clone();

    return tf.tidy(() =>
        x.expandDims(3)
            .tile([1, 1, 1, repeats, 1])
            .reshape([batch, seq, targetHeads, headDim])
    );
}

function scaledDotProductAttention(
    q,
    k,
    v,
    {
        queryPositions,
        keyPositions,
        slidingWindow = null,
    } = {}
) {
    return tf.tidy(() => {
        const qh = q.transpose([
                0,
                2,
                1,
                3
            ]);

        const kh = k.transpose([
                0,
                2,
                1,
                3
            ]);

        const vh = v.transpose([
                0,
                2,
                1,
                3
            ]);

        let logits = tf.matMul(
                qh,
                kh,
                false,
                true
            );

        const qPos = queryPositions
                .expandDims(2);

        const kPos = keyPositions
                .expandDims(1);

        let mask = kPos.lessEqual(
                qPos
            );

        if (
            slidingWindow != null
        ) {
            mask = mask
                    .logicalAnd(
                        kPos.greater(
                            qPos.sub(
                                slidingWindow
                            )
                        )
                    )
                    .logicalAnd(
                        kPos.less(
                            qPos.add(
                                slidingWindow
                            )
                        )
                    );
        }

        logits = tf.where(
                mask.expandDims(1),
                logits,
                tf.fill(
                    logits.shape,
                    -1e30
                )
            );

        const probs = tf.softmax(
                logits,
                -1
            );

        return tf.matMul(
            probs,
            vh
        )
            .transpose([
                0,
                2,
                1,
                3
            ]);
    });
}

function appendKvCache(
    cache,
    kNew,
    vNew,
    positionNew
) {
    if (!cache) {
        throw new Error(
            "KV cache object is required."
        );
    }

    const oldK = cache.k ?? null;

    const oldV = cache.v ?? null;

    const oldPositions = cache.positions ?? null;

    const nextK = tf.keep(
            oldK
                ? tf.concat(
                    [
                        oldK,
                        kNew
                    ],
                    1
                )
                : kNew.clone()
        );

    const nextV = tf.keep(
            oldV
                ? tf.concat(
                    [
                        oldV,
                        vNew
                    ],
                    1
                )
                : vNew.clone()
        );

    const nextPositions = tf.keep(
            oldPositions
                ? tf.concat(
                    [
                        oldPositions,
                        positionNew
                    ],
                    1
                )
                : positionNew.clone()
        );

    oldK?.dispose();
    oldV?.dispose();
    oldPositions?.dispose();

    cache.k = nextK;

    cache.v = nextV;

    cache.positions = nextPositions;

    return cache;
}

export function disposeKvCache(
    cache
) {
    if (!cache) {
        return;
    }

    cache.k?.dispose?.();
    cache.v?.dispose?.();
    cache.positions?.dispose?.();

    cache.k = null;
    cache.v = null;
    cache.positions = null;
}

export class OfficialW4A16Attention {
    constructor(
        config,
        { groupSize = 32, outputChunkSize = 256 } = {}
    ) {
        this.hiddenSize = config.hiddenSize;
        this.numHeads = config.numAttentionHeads;
        this.numKvHeads = config.numKeyValueHeads;
        this.headDim = config.headDim;
        this.attentionType = config.attentionType;
        this.slidingWindow = config.slidingWindow;
        this.ropeTheta = config.ropeTheta;
        this.ropeProportion = config.ropeProportion;
        this.rmsNormEps = config.rmsNormEps;
        this.isKvSharedLayer = Boolean(config.isKvSharedLayer);
        this.storeFullLengthKv = Boolean(config.storeFullLengthKv);

        this.qNorm = new RMSNorm(this.headDim, config.rmsNormEps);
        this.kNorm = new RMSNorm(this.headDim, config.rmsNormEps);

        const opts = { groupSize, outputChunkSize };

        this.qProj = new OfficialW4A16Linear(
            this.hiddenSize,
            this.numHeads * this.headDim,
            opts
        );

        this.kProj = new OfficialW4A16Linear(
            this.hiddenSize,
            this.numKvHeads * this.headDim,
            opts
        );

        this.vProj = new OfficialW4A16Linear(
            this.hiddenSize,
            this.numKvHeads * this.headDim,
            opts
        );

        this.oProj = new OfficialW4A16Linear(
            this.numHeads * this.headDim,
            this.hiddenSize,
            opts
        );
    }

    apply(
        x,
        positions,
        {
            sharedKvStates = null,
            kvCache = null,
            useKvCache = false,
        } = {}
    ) {
        return tf.tidy(() => {
            const [
                batch,
                seq
            ] = x.shape;

            let q = this.qProj
                    .apply(x)
                    .reshape([
                        batch,
                        seq,
                        this.numHeads,
                        this.headDim
                    ]);

            q = this.qNorm
                    .apply(q);

            q = applyRoPE(
                    q,
                    positions,
                    {
                        baseFrequency:
                            this.ropeTheta,
                        ropeProportion:
                            this.ropeProportion,
                    }
                );

            let k;
            let v;
            let keyPositions;

            if (
                this.isKvSharedLayer
            ) {
                if (
                    !sharedKvStates ||
                    !sharedKvStates[
                        this.attentionType
                    ]
                ) {
                    throw new Error(
                        `Missing shared KV states for ${this.attentionType}`
                    );
                }

                const shared = sharedKvStates[
                        this.attentionType
                    ];

                k = shared.k;

                v = shared.v;

                keyPositions = shared.positions;
            } else {
                let kNew = this.kProj
                        .apply(x)
                        .reshape([
                            batch,
                            seq,
                            this.numKvHeads,
                            this.headDim
                        ]);

                let vNew = this.vProj
                        .apply(x)
                        .reshape([
                            batch,
                            seq,
                            this.numKvHeads,
                            this.headDim
                        ]);

                kNew = this.kNorm
                        .apply(
                            kNew
                        );

                // Gemma 4 V normalization has no learned scale.
                vNew = vNew.mul(
                        tf.rsqrt(
                            vNew
                                .square()
                                .mean(
                                    -1,
                                    true
                                )
                                .add(
                                    this.rmsNormEps
                                )
                        )
                    );

                kNew = applyRoPE(
                        kNew,
                        positions,
                        {
                            baseFrequency:
                                this.ropeTheta,
                            ropeProportion:
                                this.ropeProportion,
                        }
                    );

                if (
                    useKvCache
                ) {
                    if (!kvCache) {
                        throw new Error(
                            "useKvCache=true requires kvCache."
                        );
                    }

                    appendKvCache(
                        kvCache,
                        kNew,
                        vNew,
                        positions
                    );

                    k = kvCache.k;

                    v = kvCache.v;

                    keyPositions = kvCache.positions;

                    if (
                        this.storeFullLengthKv
                    ) {
                        if (
                            !sharedKvStates
                        ) {
                            throw new Error(
                                "sharedKvStates is required for KV sharing."
                            );
                        }

                        // Important: no clone here.
                        // The shared layer reads the exact source-layer
                        // time-direction cache object.
                        sharedKvStates[
                            this.attentionType
                        ] = kvCache;
                    }
                } else {
                    k = kNew;

                    v = vNew;

                    keyPositions = positions;

                    if (
                        this.storeFullLengthKv
                    ) {
                        if (
                            !sharedKvStates
                        ) {
                            throw new Error(
                                "sharedKvStates object is required when storing shared KV."
                            );
                        }

                        const previous = sharedKvStates[
                                this.attentionType
                            ];

                        previous?.k?.dispose?.();
                        previous?.v?.dispose?.();
                        previous?.positions?.dispose?.();

                        sharedKvStates[
                            this.attentionType
                        ] = {
                            k:
                                tf.keep(
                                    k.clone()
                                ),
                            v:
                                tf.keep(
                                    v.clone()
                                ),
                            positions:
                                tf.keep(
                                    positions.clone()
                                ),
                        };
                    }
                }
            }

            const kExpanded = repeatKVHeads(
                    k,
                    this.numHeads
                );

            const vExpanded = repeatKVHeads(
                    v,
                    this.numHeads
                );

            const encoded = scaledDotProductAttention(
                    q,
                    kExpanded,
                    vExpanded,
                    {
                        queryPositions:
                            positions,
                        keyPositions,
                        slidingWindow:
                            this.attentionType ===
                            "sliding_attention"
                                ? this.slidingWindow
                                : null,
                    }
                );

            const merged = encoded.reshape([
                    batch,
                    seq,
                    this.numHeads *
                    this.headDim
                ]);

            return this.oProj
                .apply(
                    merged
                );
        });
    }

    dispose() {
        this.qNorm.dispose();
        this.kNorm.dispose();
        this.qProj.dispose();
        this.kProj.dispose();
        this.vProj.dispose();
        this.oProj.dispose();
    }
}

export class OfficialW4A16MLP {
    constructor(
        hiddenSize,
        intermediateSize,
        options = {}
    ) {
        this.gateProj = new OfficialW4A16Linear(
            hiddenSize,
            intermediateSize,
            options
        );
        this.upProj = new OfficialW4A16Linear(
            hiddenSize,
            intermediateSize,
            options
        );
        this.downProj = new OfficialW4A16Linear(
            intermediateSize,
            hiddenSize,
            options
        );
    }

    apply(x) {
        return tf.tidy(() => {
            const gate = this.gateProj.apply(x);
            const up = this.upProj.apply(x);
            return this.downProj.apply(
                geluTanh(gate).mul(up)
            );
        });
    }

    dispose() {
        this.gateProj.dispose();
        this.upProj.dispose();
        this.downProj.dispose();
    }
}

export class OfficialW4A16Gemma4Block {
    constructor(
        config,
        { groupSize = 32, outputChunkSize = 256 } = {}
    ) {
        this.preAttentionNorm = new RMSNorm(
            config.hiddenSize,
            config.rmsNormEps
        );
        this.postAttentionNorm = new RMSNorm(
            config.hiddenSize,
            config.rmsNormEps
        );
        this.preFfwNorm = new RMSNorm(
            config.hiddenSize,
            config.rmsNormEps
        );
        this.postFfwNorm = new RMSNorm(
            config.hiddenSize,
            config.rmsNormEps
        );
        this.postPerLayerInputNorm = new RMSNorm(
            config.hiddenSize,
            config.rmsNormEps
        );

        const opts = { groupSize, outputChunkSize };

        this.attention = new OfficialW4A16Attention(config, opts);
        this.mlp = new OfficialW4A16MLP(
            config.hiddenSize,
            config.intermediateSize,
            opts
        );
        this.perLayerInputGate = new OfficialW4A16Linear(
            config.hiddenSize,
            config.perLayerInputDim,
            opts
        );
        this.perLayerProjection = new OfficialW4A16Linear(
            config.perLayerInputDim,
            config.hiddenSize,
            opts
        );

        this.skipScale = tf.scalar(1.0, "float32");
    }

    setSkipScale(scale) {
        const next = typeof scale === "number"
                ? tf.scalar(scale, "float32")
                : tf.tidy(() => scale.reshape([]).clone());

        this.skipScale.dispose();
        this.skipScale = next;
    }

    apply(
        x,
        positions,
        perLayerInput,
        {
            returnIntermediates = false,
            sharedKvStates = null,
            kvCache = null,
            useKvCache = false,
        } = {}
    ) {
        return tf.tidy(() => {
            const preAttention = this.preAttentionNorm.apply(x);
            const attentionOutput = this.postAttentionNorm.apply(
                this.attention.apply(
                    preAttention,
                    positions,
                    {
                        sharedKvStates,
                        kvCache,
                        useKvCache,
                    }
                )
            );
            const attentionResidual = x.add(attentionOutput);

            const preFfw = this.preFfwNorm.apply(attentionResidual);
            const ffwOutput = this.postFfwNorm.apply(
                this.mlp.apply(preFfw)
            );
            const ffwResidual = attentionResidual.add(ffwOutput);

            const pleGate = this.perLayerInputGate.apply(ffwResidual);
            const pleActivated = geluTanh(pleGate);
            const pleModulated = pleActivated.mul(perLayerInput);
            const pleProjected = this.perLayerProjection.apply(pleModulated);
            const pleNormalized = this.postPerLayerInputNorm.apply(pleProjected);

            const output = ffwResidual
                .add(pleNormalized)
                .mul(this.skipScale);

            if (!returnIntermediates) {
                return output;
            }

            return {
                preAttention,
                attentionOutput,
                attentionResidual,
                preFfw,
                ffwOutput,
                ffwResidual,
                pleGate,
                pleActivated,
                pleModulated,
                pleProjected,
                pleNormalized,
                output,
            };
        });
    }

    dispose() {
        this.preAttentionNorm.dispose();
        this.postAttentionNorm.dispose();
        this.preFfwNorm.dispose();
        this.postFfwNorm.dispose();
        this.postPerLayerInputNorm.dispose();

        this.attention.dispose();
        this.mlp.dispose();
        this.perLayerInputGate.dispose();
        this.perLayerProjection.dispose();
        this.skipScale.dispose();
    }
}
