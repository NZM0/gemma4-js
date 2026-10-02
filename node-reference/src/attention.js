import * as tf from "@tensorflow/tfjs";
import { Linear } from "./linear.js";
import { RMSNorm } from "./rmsnorm.js";
import { applyRoPE } from "./rope.js";
import {
    createPositionCausalMask,
    createSlidingPositionMask
} from "./mask.js";

function repeatKV(x, repeats) {
    if (repeats === 1) {
        return x.clone();
    }

    return tf.tidy(() => {
        const [b, s, k, h] = x.shape;

        return x
            .expandDims(3)
            .tile([1, 1, 1, repeats, 1])
            .reshape([b, s, k * repeats, h]);
    });
}

export class Gemma4Attention {
    constructor({
        hiddenSize,
        numAttentionHeads,
        numKeyValueHeads,
        headDim,
        attentionType = "sliding_attention",
        slidingWindow = 512,
        ropeTheta = 10000,
        ropeProportion = 1.0,
        ropeScaleFactor = 1.0,
        attnLogitSoftcap = null,
        rmsNormEps = 1e-6,
    }) {
        if (numAttentionHeads % numKeyValueHeads !== 0) {
            throw new Error("numAttentionHeads must be divisible by numKeyValueHeads");
        }

        this.hiddenSize = hiddenSize;
        this.numAttentionHeads = numAttentionHeads;
        this.numKeyValueHeads = numKeyValueHeads;
        this.headDim = headDim;
        this.numKeyValueGroups = numAttentionHeads / numKeyValueHeads;
        this.attentionType = attentionType;
        this.slidingWindow = slidingWindow;
        this.ropeTheta = ropeTheta;
        this.ropeProportion = ropeProportion;
        this.ropeScaleFactor = ropeScaleFactor;
        this.attnLogitSoftcap = attnLogitSoftcap;
        this.qProj = new Linear(hiddenSize, numAttentionHeads * headDim);
        this.kProj = new Linear(hiddenSize, numKeyValueHeads * headDim);
        this.vProj = new Linear(hiddenSize, numKeyValueHeads * headDim);
        this.oProj = new Linear(numAttentionHeads * headDim, hiddenSize);
        this.qNorm = new RMSNorm(headDim, rmsNormEps, true);
        this.kNorm = new RMSNorm(headDim, rmsNormEps, true);
        this.vNorm = new RMSNorm(headDim, rmsNormEps, false);
    }

    _projectQKV(x, positions) {
        return tf.tidy(() => {
            const [batchSize, seqLen] = x.shape;

            let q = this.qProj
                .apply(x)
                .reshape([batchSize, seqLen, this.numAttentionHeads, this.headDim]);

            let k = this.kProj
                .apply(x)
                .reshape([batchSize, seqLen, this.numKeyValueHeads, this.headDim]);

            let v = this.vProj
                .apply(x)
                .reshape([batchSize, seqLen, this.numKeyValueHeads, this.headDim]);

            q = this.qNorm.apply(q);
            k = this.kNorm.apply(k);
            v = this.vNorm.apply(v);

            q = applyRoPE(
                q,
                positions,
                {
                    baseFrequency: this.ropeTheta,
                    scaleFactor: this.ropeScaleFactor,
                    ropeProportion: this.ropeProportion,
                }
            );

            k = applyRoPE(
                k,
                positions,
                {
                    baseFrequency: this.ropeTheta,
                    scaleFactor: this.ropeScaleFactor,
                    ropeProportion: this.ropeProportion,
                }
            );

            return {
                q,
                k,
                v
            };
        });
    }

    _attend(q, k, v, queryPositions, keyPositions) {
        return tf.tidy(() => {
            const [batchSize, queryLen] = q.shape;
            const kRepeated = repeatKV(k, this.numKeyValueGroups);
            const vRepeated = repeatKV(v, this.numKeyValueGroups);
            const qh = q.transpose([0, 2, 1, 3]);
            const kh = kRepeated.transpose([0, 2, 1, 3]);
            const vh = vRepeated.transpose([0, 2, 1, 3]);

            // Gemma 4 text attention explicitly uses scale = 1.0.
            let logits = tf.matMul(qh, kh, false, true);

            if (this.attnLogitSoftcap !== null) {
                const cap = this.attnLogitSoftcap;

                logits = tf.tanh(logits.div(cap)).mul(cap);
            }

            let mask = createPositionCausalMask(queryPositions, keyPositions);

            if (this.attentionType === "sliding_attention") {
                mask = tf.logicalAnd(
                    mask,
                    createSlidingPositionMask(queryPositions, keyPositions, this.slidingWindow)
                );
            }

            const mask4d = mask.expandDims(1);
            const paddedLogits = tf.where(mask4d, logits, tf.scalar(-1e30, "float32"));
            const probs = tf.softmax(paddedLogits, -1);
            const encoded = tf.matMul(probs, vh);

            const merged = encoded
                .transpose([0, 2, 1, 3])
                .reshape([batchSize, queryLen, this.numAttentionHeads * this.headDim]);

            return this.oProj.apply(merged);
        });
    }

    /**
     * Dense Gemma 4 forward with vertical KV sharing.
     *
     * Gemma 4 E2B computes K/V normally through layer 14.
     * Layer 13 stores the sliding-attention KV state and layer 14
     * stores the full-attention KV state. Layers 15..34 reuse the
     * corresponding state instead of recomputing K/V.
     *
     * This method is intentionally separate from apply() so the
     * reference implementation remains easy to inspect.
     */
    applyWithSharedKv(
        x,
        positions,
        {
            sharedKvState = null,
            captureKv = false,
        } = {}
    ) {
        const result = tf.tidy(() => {
            const [batchSize, seqLen] = x.shape;

            let q = this.qProj
                .apply(x)
                .reshape([batchSize, seqLen, this.numAttentionHeads, this.headDim]);

            q = this.qNorm.apply(q);

            q = applyRoPE(
                q,
                positions,
                {
                    baseFrequency: this.ropeTheta,
                    scaleFactor: this.ropeScaleFactor,
                    ropeProportion: this.ropeProportion,
                }
            );

            let k;
            let v;
            let keyPositions;

            if (sharedKvState !== null) {
                k = sharedKvState.k;
                v = sharedKvState.v;
                keyPositions = sharedKvState.positions;
            } else {
                let kNew = this.kProj
                    .apply(x)
                    .reshape([batchSize, seqLen, this.numKeyValueHeads, this.headDim]);

                let vNew = this.vProj
                    .apply(x)
                    .reshape([batchSize, seqLen, this.numKeyValueHeads, this.headDim]);

                kNew = this.kNorm.apply(kNew);
                vNew = this.vNorm.apply(vNew);

                kNew = applyRoPE(
                    kNew,
                    positions,
                    {
                        baseFrequency: this.ropeTheta,
                        scaleFactor: this.ropeScaleFactor,
                        ropeProportion: this.ropeProportion,
                    }
                );

                k = kNew;
                v = vNew;
                keyPositions = positions;
            }

            const output = this._attend(q, k, v, positions, keyPositions);

            if (!captureKv) {
                    // Returning a Tensor from tf.tidy() already preserves it.
                    // Do not tf.keep() it here: this method is itself called
                    // inside the block-level tidy, and keep() would prevent the
                    // block tidy from reclaiming this intermediate after the
                    // residual path consumes it.
                return {
                    output,
                    kvState: null,
                };
            }

            return {
                output,
                kvState: {
                        // These clones are intentionally returned as persistent
                        // shared states. The surrounding TensorContainer return
                        // keeps them alive across the tidy boundary until the
                        // transformer stack disposes them explicitly.
                    k: k.clone(),
                    v: v.clone(),
                    positions: keyPositions.clone(),
                },
            };
        });

        return result;
    }

    apply(x, positions) {
        return tf.tidy(() => {
            const {
                q,
                k,
                v
            } = this._projectQKV(x, positions);

            return this._attend(q, k, v, positions, positions);
        });
    }

    /**
     * Cache-aware attention.
     *
     * Past K is already normalized + RoPE transformed.
     * Past V is already value-normalized.
     */
    applyWithCache(x, positions, pastCache = null) {
        return tf.tidy(() => {
            const {
                q,
                k: newK,
                v: newV
            } = this._projectQKV(x, positions);

            let allK = newK;
            let allV = newV;
            let allPositions = positions;

            if (pastCache) {
                allK = tf.concat([pastCache.k, newK], 1);
                allV = tf.concat([pastCache.v, newV], 1);
                allPositions = tf.concat([pastCache.positions, positions], 1);
            }

            const output = this._attend(q, allK, allV, positions, allPositions);

            // For local layers, only the newest window can ever be
            // attended by future single-token decoding queries.
            let cacheK = allK;
            let cacheV = allV;
            let cachePositions = allPositions;

            if (this.attentionType === "sliding_attention" && allK.shape[1] > this.slidingWindow) {
                const start = allK.shape[1] - this.slidingWindow;

                cacheK = allK.slice([0, start, 0, 0], [-1, this.slidingWindow, -1, -1]);
                cacheV = allV.slice([0, start, 0, 0], [-1, this.slidingWindow, -1, -1]);
                cachePositions = allPositions.slice([0, start], [-1, this.slidingWindow]);
            }

            return {
                // Returned tensors automatically survive this tidy().
                // Do not tf.keep() them here; ownership is decided by
                // the caller/model-level cache.
                output,

                cache: {
                    k: cacheK.clone(),
                    v: cacheV.clone(),
                    positions: cachePositions.clone(),
                }
            };
        });
    }

    dispose() {
        this.qProj.dispose();
        this.kProj.dispose();
        this.vProj.dispose();
        this.oProj.dispose();

        this.qNorm.dispose();
        this.kNorm.dispose();
        this.vNorm.dispose();
    }
}
