export const TFJS_WEBGL_BUILD = "0.3.0-webgl-distribution";

const tf =
    globalThis.tf;

import {
    BrowserW4A16Reader
} from "./checkpoint/browser-w4a16-reader.js?v=0.2.2";

import {
    createOfficialW4A16E2BBlock,
    getOfficialW4A16LayerSpec
} from "./official-w4a16-layer-factory.js?v=0.2.2";

import {
    loadOfficialW4A16Layer
} from "./checkpoint/load-official-w4a16-layer.js?v=0.2.2";

import {
    OfficialW4A16Linear
} from "./official-quantized-linear.js?v=0.2.2";

import {
    disposeKvCache
} from "./official-w4a16-layer0.js?v=0.2.2";

function rms(
    x,
    scale,
    eps = 1e-6
) {
    return tf.tidy(
        () => {
            const y =
                x.mul(
                    tf.rsqrt(
                        x.square()
                            .mean(
                                -1,
                                true
                            )
                            .add(
                                eps
                            )
                    )
                );

            return scale
                ? y.mul(
                    scale
                )
                : y;
        }
    );
}

function topK(
    logits,
    k = 10
) {
    const best =
        [];

    for (
        let i = 0;
        i < logits.length;
        i++
    ) {
        const value =
            logits[i];

        if (
            !Number.isFinite(
                value
            )
        ) {
            continue;
        }

        if (
            best.length <
                k
            ||
            value >
                best[
                    best.length -
                    1
                ].value
        ) {
            best.push({
                tokenId:
                    i,
                value,
            });

            best.sort(
                (
                    a,
                    b
                ) =>
                    b.value -
                    a.value
            );

            if (
                best.length >
                k
            ) {
                best.pop();
            }
        }
    }

    return best;
}

export class TfjsGemma4Backend {
    constructor(
        modelFile,
        {
            backend =
                "webgl",
            maxSeq =
                512,
            outputChunkSize =
                512,
            lmChunkRows =
                2048,
            log =
                () => {},
            pleRowProvider =
                null,
        } = {}
    ) {
        this.modelFile =
            modelFile;

        this.backend =
            backend;

        this.maxSeq =
            maxSeq;

        this.outputChunkSize =
            outputChunkSize;

        this.lmChunkRows =
            lmChunkRows;

        this.log =
            log;

        this.pleRowProvider =
            pleRowProvider;

        this.reader =
            null;

        this.blocks =
            [];

        this.pleProjection =
            null;

        this.pleNorm =
            null;

        this.finalNorm =
            null;

        this.lmHeadChunks =
            [];

        this.layerKvCaches =
            [];

        this.sharedKvStates =
            {};

        this.vocab =
            0;

        this.names = {
            embed:
                "model.language_model.embed_tokens.weight",
            ple:
                "model.language_model.embed_tokens_per_layer.weight",
            projBase:
                "model.language_model.per_layer_model_projection",
            pleNorm:
                "model.language_model.per_layer_projection_norm.weight",
            finalNorm:
                "model.language_model.norm.weight",
        };
    }

    async load() {
        this.log(
            `Switching TensorFlow.js backend -> ${this.backend}`
        );

        await tf.setBackend(
            this.backend
        );

        await tf.ready();

        this.log(
            `TensorFlow.js active backend: ${tf.getBackend()}`
        );

        this.reader =
            await new BrowserW4A16Reader(
                this.modelFile
            )
                .open();

        this.vocab =
            this.reader.info(
                this.names.embed
            ).shape[0];

        this.pleProjection =
            await this.loadTopLevelLinear(
                this.names.projBase,
                1536,
                35 * 256
            );

        this.pleNorm =
            await this.tensor(
                this.names.pleNorm
            );

        this.finalNorm =
            await this.tensor(
                this.names.finalNorm
            );

        this.log(
            "Loading 35 TF.js reference blocks (packed W4A16 stays packed in JS memory)..."
        );

        for (
            let layerIndex = 0;
            layerIndex < 35;
            layerIndex++
        ) {
            const spec =
                getOfficialW4A16LayerSpec(
                    layerIndex
                );

            this.log(
                `TF.js Layer ${String(layerIndex).padStart(2, "0")} ` +
                `${spec.full ? "GLOBAL" : "local"}...`
            );

            const block =
                createOfficialW4A16E2BBlock(
                    layerIndex,
                    {
                        outputChunkSize:
                            this.outputChunkSize,
                    }
                );

            await loadOfficialW4A16Layer(
                block,
                layerIndex,
                this.reader
            );

            this.blocks.push(
                block
            );
        }

        this.layerKvCaches =
            Array.from(
                {
                    length:
                        35,
                },
                (
                    _,
                    i
                ) =>
                    getOfficialW4A16LayerSpec(
                        i
                    ).isKvSharedLayer
                        ? null
                        : {
                            k:
                                null,
                            v:
                                null,
                            positions:
                                null,
                        }
            );

        this.log(
            `Preloading tied LM head into ${this.backend} as ` +
            `${this.lmChunkRows}-row float32 chunks...`
        );

        for (
            let start = 0;
            start < this.vocab;
            start += this.lmChunkRows
        ) {
            const count =
                Math.min(
                    this.lmChunkRows,
                    this.vocab -
                    start
                );

            const rows =
                await this.reader
                    .readBF16RowRange(
                        this.names.embed,
                        start,
                        count
                    );

            const tensor =
                tf.tensor2d(
                    rows.values,
                    rows.shape,
                    "float32"
                );

            this.lmHeadChunks.push({
                start,
                count,
                tensor,
            });

            if (
                start %
                32768 ===
                0
            ) {
                this.log(
                    `  TF.js LM head vocab ${start}..${start + count - 1}`
                );
            }
        }

        this.log(
            `TF.js ${this.backend} preload complete.`
        );

        this.log(
            "tf.memory():",
            JSON.stringify(
                tf.memory()
            )
        );
    }

    hasTensor(
        name
    ) {
        return Object.prototype
            .hasOwnProperty
            .call(
                this.reader.header,
                name
            );
    }

    async tensor(
        name
    ) {
        const x =
            await this.reader
                .readBF16Float32(
                    name
                );

        return tf.tensor(
            x.values,
            x.shape,
            "float32"
        );
    }

    async loadTopLevelLinear(
        base,
        inDim,
        outDim
    ) {
        const packedName =
            `${base}.weight_packed`;

        const denseName =
            `${base}.weight`;

        if (
            this.hasTensor(
                packedName
            )
        ) {
            const packed =
                await this.reader
                    .readInt32(
                        packedName
                    );

            const scale =
                await this.reader
                    .readBF16Float32(
                        `${base}.weight_scale`
                    );

            const linear =
                new OfficialW4A16Linear(
                    inDim,
                    outDim,
                    {
                        groupSize:
                            32,
                        outputChunkSize:
                            this.outputChunkSize,
                    }
                );

            linear.setOfficialWeights({
                packedWeight:
                    packed.values,
                scales:
                    scale.values,
            });

            return linear;
        }

        if (
            this.hasTensor(
                denseName
            )
        ) {
            const dense =
                await this.tensor(
                    denseName
                );

            return {
                apply:
                    x =>
                        tf.matMul(
                            x,
                            dense,
                            false,
                            true
                        ),

                dispose:
                    () =>
                        dense.dispose(),
            };
        }

        throw new Error(
            `No supported representation for ${base}`
        );
    }

    async rows(
        name,
        ids
    ) {
        const result =
            await this.reader
                .readBF16Rows(
                    name,
                    ids
                );

        return tf.tensor2d(
            result.values,
            result.shape,
            "float32"
        );
    }

    async buildInputFeatures(
        tokenIds
    ) {
        const seq =
            tokenIds.length;

        const emb =
            await this.rows(
                this.names.embed,
                tokenIds
            );

        const pleRows =
            this.pleRowProvider
                ? tf.tensor2d(
                    (
                        await this.pleRowProvider
                            .readRows(
                                tokenIds
                            )
                    ).values,
                    [
                        tokenIds.length,
                        8960,
                    ],
                    "float32"
                )
                : await this.rows(
                    this.names.ple,
                    tokenIds
                );

        const hidden =
            tf.tidy(
                () =>
                    emb
                        .reshape([
                            1,
                            seq,
                            1536,
                        ])
                        .mul(
                            Math.sqrt(
                                1536
                            )
                        )
            );

        const perLayer =
            tf.tidy(
                () => {
                    const projected =
                        this.pleProjection
                            .apply(
                                hidden
                            )
                            .mul(
                                1 /
                                Math.sqrt(
                                    1536
                                )
                            )
                            .reshape([
                                1,
                                seq,
                                35,
                                256,
                            ]);

                    const normalized =
                        rms(
                            projected,
                            this.pleNorm
                        );

                    const tokenPle =
                        pleRows
                            .reshape([
                                1,
                                seq,
                                35,
                                256,
                            ])
                            .mul(
                                Math.sqrt(
                                    256
                                )
                            );

                    return normalized
                        .add(
                            tokenPle
                        )
                        .mul(
                            1 /
                            Math.sqrt(
                                2
                            )
                        );
                }
            );

        emb.dispose();
        pleRows.dispose();

        return {
            hidden,
            perLayer,
        };
    }

    async runCachedForward(
        tokenIds,
        absoluteStartPosition
    ) {
        const seq =
            tokenIds.length;

        const {
            hidden:
                initialHidden,
            perLayer,
        } =
            await this.buildInputFeatures(
                tokenIds
            );

        let hidden =
            initialHidden;

        const positions =
            tf.tensor2d(
                [
                    Array.from(
                        {
                            length:
                                seq,
                        },
                        (
                            _,
                            i
                        ) =>
                            absoluteStartPosition +
                            i
                    )
                ],
                [
                    1,
                    seq,
                ],
                "int32"
            );

        try {
            for (
                let layerIndex = 0;
                layerIndex < 35;
                layerIndex++
            ) {
                const ple =
                    tf.tidy(
                        () =>
                            perLayer
                                .slice(
                                    [
                                        0,
                                        0,
                                        layerIndex,
                                        0,
                                    ],
                                    [
                                        1,
                                        seq,
                                        1,
                                        256,
                                    ]
                                )
                                .squeeze(
                                    [
                                        2
                                    ]
                                )
                    );

                const next =
                    this.blocks[
                        layerIndex
                    ]
                        .apply(
                            hidden,
                            positions,
                            ple,
                            {
                                sharedKvStates:
                                    this.sharedKvStates,
                                kvCache:
                                    this.layerKvCaches[
                                        layerIndex
                                    ],
                                useKvCache:
                                    true,
                            }
                        );

                hidden.dispose();

                hidden =
                    next;

                ple.dispose();
            }

            const lastHidden =
                tf.tidy(
                    () =>
                        rms(
                            hidden,
                            this.finalNorm
                        )
                            .slice(
                                [
                                    0,
                                    seq - 1,
                                    0,
                                ],
                                [
                                    1,
                                    1,
                                    1536,
                                ]
                            )
                            .reshape([
                                1,
                                1536,
                            ])
                );

            return lastHidden;
        } finally {
            hidden.dispose();
            perLayer.dispose();
            positions.dispose();
        }
    }

    async computeTiedHeadLogits(
        lastHidden
    ) {
        const chunks =
            [];

        try {
            for (
                const item
                of
                this.lmHeadChunks
            ) {
                const out =
                    tf.tidy(
                        () => {
                            const raw =
                                tf.matMul(
                                    lastHidden,
                                    item.tensor,
                                    false,
                                    true
                                )
                                    .div(
                                        Math.sqrt(
                                            1536
                                        )
                                    );

                            return raw
                                .div(
                                    30
                                )
                                .tanh()
                                .mul(
                                    30
                                );
                        }
                    );

                chunks.push(
                    out
                );
            }

            const joined =
                tf.concat(
                    chunks,
                    -1
                );

            const values =
                await joined.data();

            joined.dispose();

            return new Float32Array(
                values
            );
        } finally {
            for (
                const chunk
                of
                chunks
            ) {
                chunk.dispose();
            }
        }
    }

    async greedy(
        tokenIds,
        startPos
    ) {
        const hidden =
            await this.runCachedForward(
                tokenIds,
                startPos
            );

        try {
            const logits =
                await this.computeTiedHeadLogits(
                    hidden
                );

            const top10 =
                topK(
                    logits,
                    10
                );

            if (
                top10.length ===
                0
            ) {
                throw new Error(
                    "TF.js backend produced no finite logits."
                );
            }

            return {
                logits,
                top10,
                nextTokenId:
                    top10[0]
                        .tokenId,
            };
        } finally {
            hidden.dispose();
        }
    }

    resetCache() {
        for (
            const cache
            of
            this.layerKvCaches
        ) {
            disposeKvCache(
                cache
            );
        }

        this.sharedKvStates =
            {};
    }

    cacheSummary() {
        let caches =
            0;

        let bytes =
            0;

        for (
            const cache
            of
            this.layerKvCaches
        ) {
            if (
                !cache
            ) {
                continue;
            }

            if (
                cache.k
                ||
                cache.v
                ||
                cache.positions
            ) {
                caches++;
            }

            for (
                const tensor
                of
                [
                    cache.k,
                    cache.v,
                    cache.positions,
                ]
            ) {
                if (
                    tensor
                ) {
                    bytes +=
                        tensor.size *
                        4;
                }
            }
        }

        return {
            caches,
            bytes,
        };
    }

    memorySummary() {
        return tf.memory();
    }

    dispose() {
        this.resetCache();

        for (
            const block
            of
            this.blocks
        ) {
            block.dispose();
        }

        this.blocks =
            [];

        this.pleProjection
            ?.dispose();

        this.pleProjection =
            null;

        this.pleNorm
            ?.dispose();

        this.pleNorm =
            null;

        this.finalNorm
            ?.dispose();

        this.finalNorm =
            null;

        for (
            const item
            of
            this.lmHeadChunks
        ) {
            item.tensor.dispose();
        }

        this.lmHeadChunks =
            [];

        this.reader
            ?.close();

        this.reader =
            null;
    }
}
