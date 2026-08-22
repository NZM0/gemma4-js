import * as tf from "@tensorflow/tfjs";

import {
    NodeSafeTensorsReader
} from "./checkpoint/node-safetensors-reader.js";

import {
    getRealLayerSpec,
    loadRealGemma4Layer
} from "./checkpoint/load-real-layer.js";

import {
    createRealE2BBlock
} from "./real-layer-factory.js";

const EMBEDDING_NAME =
    "model.language_model.embed_tokens.weight";

const PLE_TABLE_NAME =
    "model.language_model.embed_tokens_per_layer.weight";

const PLE_PROJECTION_NAME =
    "model.language_model.per_layer_model_projection.weight";

const PLE_NORM_NAME =
    "model.language_model.per_layer_projection_norm.weight";

const FINAL_NORM_NAME =
    "model.language_model.norm.weight";

const HIDDEN_SIZE = 1536;
const NUM_LAYERS = 35;
const PLE_DIM = 256;
const VOCAB_SIZE = 262144;
const FINAL_LOGIT_SOFTCAP = 30.0;

function rmsNorm(
    x,
    scale,
    eps = 1e-6
) {
    return tf.tidy(() => {
        const meanSquare =
            x.square()
                .mean(
                    -1,
                    true
                );

        return x
            .mul(
                tf.rsqrt(
                    meanSquare.add(eps)
                )
            )
            .mul(scale);
    });
}

export async function prepareRealInputAndPle(
    reader,
    tokenIds
) {
    const seqLen =
        tokenIds.length;

    const embeddingRows =
        await reader.readRowsTensor(
            EMBEDDING_NAME,
            tokenIds
        );

    const pleRows =
        await reader.readRowsTensor(
            PLE_TABLE_NAME,
            tokenIds
        );

    const projection =
        await reader.readTensor(
            PLE_PROJECTION_NAME
        );

    const projectionNorm =
        await reader.readTensor(
            PLE_NORM_NAME
        );

    const result =
        tf.tidy(() => {
            const hidden =
                embeddingRows
                    .reshape([
                        1,
                        seqLen,
                        HIDDEN_SIZE
                    ])
                    .mul(
                        Math.sqrt(
                            HIDDEN_SIZE
                        )
                    );

            // Checkpoint projection layout is [8960,1536].
            // Use transposeB=true directly so no 55MB transposed
            // parameter needs to be kept.
            let context =
                tf.matMul(
                    hidden,
                    projection,
                    false,
                    true
                )
                .mul(
                    1.0 /
                    Math.sqrt(
                        HIDDEN_SIZE
                    )
                )
                .reshape([
                    1,
                    seqLen,
                    NUM_LAYERS,
                    PLE_DIM
                ]);

            context =
                rmsNorm(
                    context,
                    projectionNorm
                );

            const tokenIdentity =
                pleRows
                    .reshape([
                        1,
                        seqLen,
                        NUM_LAYERS,
                        PLE_DIM
                    ])
                    .mul(
                        Math.sqrt(
                            PLE_DIM
                        )
                    );

            const perLayerInputs =
                context
                    .add(
                        tokenIdentity
                    )
                    .mul(
                        1.0 /
                        Math.sqrt(2.0)
                    );

            return {
                hidden,
                perLayerInputs,
            };
        });

    embeddingRows.dispose();
    pleRows.dispose();
    projection.dispose();
    projectionNorm.dispose();

    return result;
}

export async function runRealTransformerStack(
    reader,
    safetensorsPath,
    header,
    hiddenInput,
    perLayerInputs,
    positions,
    {
        onLayer = null
    } = {}
) {
    let hidden =
        hiddenInput;

    const sharedKvStates = {
        sliding_attention:
            null,
        full_attention:
            null,
    };

    try {
    for (
        let layerIndex = 0;
        layerIndex < NUM_LAYERS;
        layerIndex++
    ) {
        const block =
            createRealE2BBlock(
                layerIndex
            );

        await loadRealGemma4Layer(
            block,
            layerIndex,
            {
                safetensorsPath,
                header,
                reader,
            }
        );

        const ple =
            tf.tidy(() => {
                return perLayerInputs
                    .slice(
                        [
                            0,
                            0,
                            layerIndex,
                            0
                        ],
                        [
                            -1,
                            -1,
                            1,
                            PLE_DIM
                        ]
                    )
                    .squeeze([2]);
            });

        const spec =
            getRealLayerSpec(
                layerIndex
            );

        const sharedKvState =
            spec.isKvSharedLayer
                ? sharedKvStates[
                    spec.attentionType
                ]
                : null;

        if (
            spec.isKvSharedLayer &&
            sharedKvState == null
        ) {
            throw new Error(
                `Missing shared KV state for ${spec.attentionType} at layer ${layerIndex}.`
            );
        }

        const result =
            block.applyWithSharedKv(
                hidden,
                positions,
                ple,
                {
                    sharedKvState,
                    captureKv:
                        spec.captureSharedKv,
                }
            );

        if (
            spec.captureSharedKv
        ) {
            const previous =
                sharedKvStates[
                    spec.attentionType
                ];

            previous?.k?.dispose();
            previous?.v?.dispose();
            previous?.positions?.dispose();

            sharedKvStates[
                spec.attentionType
            ] =
                result.kvState;
        }

        if (hidden !== hiddenInput) {
            hidden.dispose();
        }

        ple.dispose();
        block.dispose();

        hidden =
            result.output;

        if (onLayer) {
            await onLayer(
                layerIndex,
                hidden
            );
        }
    }

    return hidden;
    } finally {
        for (
            const state
            of
            Object.values(
                sharedKvStates
            )
        ) {
            state?.k?.dispose();
            state?.v?.dispose();
            state?.positions?.dispose();
        }
    }
}

export async function applyRealFinalNorm(
    reader,
    hidden
) {
    const scale =
        await reader.readTensor(
            FINAL_NORM_NAME
        );

    const output =
        rmsNorm(
            hidden,
            scale
        );

    scale.dispose();

    return output;
}

function applyFinalLogitSoftcap(
    logits
) {
    return tf.tidy(() => {
        return tf
            .tanh(
                logits.div(
                    FINAL_LOGIT_SOFTCAP
                )
            )
            .mul(
                FINAL_LOGIT_SOFTCAP
            );
    });
}

/**
 * Compute the entire [B,T,262144] LM head in chunks without
 * keeping the embedding table resident in float32.
 *
 * `onChunk` receives a Float32Array and can compare/store it.
 */
export async function streamFullVocabularyLogits(
    reader,
    finalHidden,
    {
        chunkRows = 2048,
        onChunk = null,
    } = {}
) {
    const [
        batchSize,
        seqLen,
        hiddenSize
    ] = finalHidden.shape;

    if (
        batchSize !== 1 ||
        hiddenSize !== HIDDEN_SIZE
    ) {
        throw new Error(
            `Expected finalHidden [1,T,1536], got [${finalHidden.shape}]`
        );
    }

    for (
        let start = 0;
        start < VOCAB_SIZE;
        start += chunkRows
    ) {
        const count =
            Math.min(
                chunkRows,
                VOCAB_SIZE - start
            );

        const rows =
            await reader.readRowRangeTensor(
                EMBEDDING_NAME,
                start,
                count
            );

        const logits =
            tf.tidy(() => {
                const raw =
                    tf.matMul(
                        finalHidden,
                        rows,
                        false,
                        true
                    );

                return applyFinalLogitSoftcap(
                    raw
                );
            });

        const values =
            await logits.data();

        if (onChunk) {
            await onChunk({
                start,
                count,
                batchSize,
                seqLen,
                values,
            });
        }

        rows.dispose();
        logits.dispose();
    }
}


/**
 * Scan the tied LM head for the LAST sequence position only.
 *
 * This avoids materializing [T, vocab] during autoregressive generation.
 * It still scans all 262144 vocabulary rows in contiguous chunks.
 */
export async function selectNextTokenFromLastPosition(
    reader,
    finalHidden,
    {
        chunkRows = 2048,
        temperature = 0.0,
        topK = 1,
        random = Math.random,
    } = {}
) {
    const [
        batchSize,
        seqLen,
        hiddenSize
    ] = finalHidden.shape;

    if (
        batchSize !== 1 ||
        hiddenSize !== HIDDEN_SIZE
    ) {
        throw new Error(
            `Expected finalHidden [1,T,1536], got [${finalHidden.shape}]`
        );
    }

    const lastHidden =
        tf.tidy(() => {
            return finalHidden
                .slice(
                    [0, seqLen - 1, 0],
                    [1, 1, HIDDEN_SIZE]
                )
                .reshape(
                    [1, HIDDEN_SIZE]
                );
        });

    // Keep only the best candidates globally while scanning chunks.
    const candidateCount =
        Math.max(
            1,
            Math.min(
                Number.isInteger(topK)
                    ? topK
                    : 1,
                VOCAB_SIZE
            )
        );

    let candidates = [];

    try {
        for (
            let start = 0;
            start < VOCAB_SIZE;
            start += chunkRows
        ) {
            const count =
                Math.min(
                    chunkRows,
                    VOCAB_SIZE - start
                );

            const rows =
                await reader.readRowRangeTensor(
                    EMBEDDING_NAME,
                    start,
                    count
                );

            const logits =
                tf.tidy(() => {
                    const raw =
                        tf.matMul(
                            lastHidden,
                            rows,
                            false,
                            true
                        )
                        .reshape([count]);

                    return applyFinalLogitSoftcap(
                        raw
                    );
                });

            const values =
                await logits.data();

            for (
                let j = 0;
                j < count;
                j++
            ) {
                candidates.push({
                    tokenId: start + j,
                    logit: values[j],
                });
            }

            candidates.sort(
                (a, b) =>
                    b.logit -
                    a.logit
            );

            if (
                candidates.length >
                candidateCount
            ) {
                candidates.length =
                    candidateCount;
            }

            rows.dispose();
            logits.dispose();
        }
    } finally {
        lastHidden.dispose();
    }

    if (
        temperature == null ||
        temperature <= 0 ||
        candidateCount === 1
    ) {
        return {
            tokenId:
                candidates[0].tokenId,

            logit:
                candidates[0].logit,

            candidates,
        };
    }

    const scaled =
        candidates.map(
            item =>
                item.logit /
                temperature
        );

    const maxLogit =
        Math.max(
            ...scaled
        );

    const weights =
        scaled.map(
            value =>
                Math.exp(
                    value -
                    maxLogit
                )
        );

    const total =
        weights.reduce(
            (sum, value) =>
                sum + value,
            0
        );

    let threshold =
        random() *
        total;

    let selected =
        candidates[
            candidates.length - 1
        ];

    for (
        let i = 0;
        i < candidates.length;
        i++
    ) {
        threshold -=
            weights[i];

        if (
            threshold <= 0
        ) {
            selected =
                candidates[i];

            break;
        }
    }

    return {
        tokenId:
            selected.tokenId,

        logit:
            selected.logit,

        candidates,
    };
}

export const REAL_FULL_FORWARD_CONSTANTS = {
    hiddenSize: HIDDEN_SIZE,
    numLayers: NUM_LAYERS,
    pleDim: PLE_DIM,
    vocabSize: VOCAB_SIZE,
    finalLogitSoftcap:
        FINAL_LOGIT_SOFTCAP,
};
