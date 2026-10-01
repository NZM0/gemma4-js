import * as tf from "@tensorflow/tfjs";
import { RMSNorm } from "./rmsnorm.js";
import { Linear } from "./linear.js";
import { GELUPytorchTanh, Gemma4MLP } from "./mlp.js";
import { Gemma4Attention } from "./attention.js";

/**
 * Gemma 4 dense Transformer block.
 *
 * Ported from google-deepmind/gemma:
 * gemma/gm/nn/gemma4/_modules.py::Block
 *
 * Forward order:
 *
 * 1. pre_attention_norm(x)
 * 2. attention(...)
 * 3. optional post_attention_norm
 * 4. + residual x
 * 5. pre_ffw_norm
 * 6. MLP
 * 7. optional post_ffw_norm
 * 8. + attention residual stream
 * 9. PLE gate -> GELU -> * per_layer_input -> projection -> RMSNorm
 * 10. + PLE residual
 * 11. * skip_scale
 *
 */
export class Gemma4Block {
    constructor({
        hiddenSize,
        intermediateSize,
        numAttentionHeads,
        numKeyValueHeads,
        headDim,
        attentionType,
        slidingWindow,
        ropeTheta,
        ropeProportion = 1.0,
        ropeScaleFactor = 1.0,
        attnLogitSoftcap = null,
        rmsNormEps = 1e-6,
        perLayerInputDim = 0,
        usePostAttentionNorm = true,
        usePostFfwNorm = true,
    }) {
        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;
        this.perLayerInputDim = perLayerInputDim;
        this.usePostAttentionNorm = usePostAttentionNorm;
        this.usePostFfwNorm = usePostFfwNorm;

        this.preAttentionNorm = new RMSNorm(
            hiddenSize,
            rmsNormEps,
            true
        );

        this.attention = new Gemma4Attention({
            hiddenSize,
            numAttentionHeads,
            numKeyValueHeads,
            headDim,
            attentionType,
            slidingWindow,
            ropeTheta,
            ropeProportion,
            ropeScaleFactor,
            attnLogitSoftcap,
            rmsNormEps,
        });

        this.postAttentionNorm = usePostAttentionNorm
            ? new RMSNorm(hiddenSize, rmsNormEps, true)
            : null;

        this.preFfwNorm = new RMSNorm(
            hiddenSize,
            rmsNormEps,
            true
        );

        this.mlp = new Gemma4MLP(
            hiddenSize,
            intermediateSize
        );

        this.postFfwNorm = usePostFfwNorm
            ? new RMSNorm(hiddenSize, rmsNormEps, true)
            : null;

        if (perLayerInputDim > 0) {
            this.perLayerInputGate = new Linear(
                hiddenSize,
                perLayerInputDim
            );

            this.perLayerProjection = new Linear(
                perLayerInputDim,
                hiddenSize
            );

            this.postPerLayerInputNorm = new RMSNorm(
                hiddenSize,
                rmsNormEps,
                true
            );

            this.pleActivation = new GELUPytorchTanh();
        } else {
            this.perLayerInputGate = null;
            this.perLayerProjection = null;
            this.postPerLayerInputNorm = null;
            this.pleActivation = null;
        }

        // Official JAX block has a learned scalar skip_scale,
        // initialized to 1.0.
        this.skipScale = tf.scalar(1.0, "float32");
    }

    setSkipScale(scale) {
        let next;

        if (typeof scale === "number") {
            next = tf.scalar(scale, "float32");
        } else {
            if (scale.size !== 1) {
                throw new Error(
                    `skipScale must contain one value, got shape [${scale.shape}]`
                );
            }

            // reshape() creates a temporary Tensor object in TF.js.
            // Build the scalar inside tidy so only the cloned persistent
            // parameter survives and the reshape temporary is disposed.
            next = tf.tidy(() => {
                return scale
                    .reshape([])
                    .clone();
            });
        }

        this.skipScale.dispose();
        this.skipScale = next;
    }

    /**
     * @param {tf.Tensor} x              [B, T, hiddenSize]
     * @param {tf.Tensor} positions      [B, T]
     * @param {tf.Tensor|null} perLayerInput [B, T, perLayerInputDim]
     * @param {object} options
     * @param {boolean} options.returnIntermediates
     *
     * returnIntermediates=true is intended for Step 4 numerical comparison.
     */
    apply(
        x,
        positions,
        perLayerInput = null,
        { returnIntermediates = false } = {}
    ) {
        if (
            x.shape.length !== 3 ||
            x.shape[2] !== this.hiddenSize
        ) {
            throw new Error(
                `x must be [B,T,${this.hiddenSize}], got [${x.shape}]`
            );
        }

        if (this.perLayerInputDim > 0) {
            if (perLayerInput === null) {
                throw new Error(
                    "perLayerInput is required when perLayerInputDim > 0"
                );
            }

            if (
                perLayerInput.shape.length !== 3 ||
                perLayerInput.shape[0] !== x.shape[0] ||
                perLayerInput.shape[1] !== x.shape[1] ||
                perLayerInput.shape[2] !== this.perLayerInputDim
            ) {
                throw new Error(
                    `perLayerInput must be ` +
                    `[${x.shape[0]},${x.shape[1]},${this.perLayerInputDim}], ` +
                    `got [${perLayerInput.shape}]`
                );
            }
        }

        if (!returnIntermediates) {
            return tf.tidy(() => {
                return this._forward(x, positions, perLayerInput).output;
            });
        }

        // Debug path: keep tensors so the caller can inspect them.
        // The caller must dispose every returned tensor.
        return this._forwardWithKeptIntermediates(
            x,
            positions,
            perLayerInput
        );
    }

    _forward(x, positions, perLayerInput) {
        const preAttention = this.preAttentionNorm.apply(x);

        let attentionOutput = this.attention.apply(
            preAttention,
            positions
        );

        if (this.postAttentionNorm) {
            attentionOutput = this.postAttentionNorm.apply(attentionOutput);
        }

        const attentionResidual = attentionOutput.add(x);

        const preFfw = this.preFfwNorm.apply(attentionResidual);

        let ffwOutput = this.mlp.apply(preFfw);

        if (this.postFfwNorm) {
            ffwOutput = this.postFfwNorm.apply(ffwOutput);
        }

        const ffwResidual = ffwOutput.add(attentionResidual);

        let output = ffwResidual;

        let pleGate = null;
        let pleActivated = null;
        let pleModulated = null;
        let pleProjected = null;
        let pleNormalized = null;

        if (this.perLayerInputDim > 0) {
            pleGate = this.perLayerInputGate.apply(output);

            pleActivated = this.pleActivation.apply(pleGate);

            pleModulated = pleActivated.mul(perLayerInput);

            pleProjected = this.perLayerProjection.apply(pleModulated);

            pleNormalized = this.postPerLayerInputNorm.apply(pleProjected);

            output = output.add(pleNormalized);
        }

        const scaledOutput = output.mul(this.skipScale);

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
            output: scaledOutput,
        };
    }

    _forwardWithKeptIntermediates(
        x,
        positions,
        perLayerInput
    ) {
        return tf.tidy(() => {
            const result = this._forward(x, positions, perLayerInput);

            const kept = {};

            for (const [name, tensor] of Object.entries(result)) {
                kept[name] = tensor === null ? null : tf.keep(tensor);
            }

            return kept;
        });
    }


    /**
     * Reference forward used by the real dense checkpoint runtime.
     *
     * `sharedKvState` is supplied only for Gemma 4's vertically
     * KV-shared layers. `captureKv=true` is used at the two source
     * layers that seed sliding/full shared KV states.
     */
    applyWithSharedKv(
        x,
        positions,
        perLayerInput,
        {
            sharedKvState = null,
            captureKv = false,
        } = {}
    ) {
        return tf.tidy(() => {
            const preAttention = this.preAttentionNorm
                    .apply(x);

            const attention = this.attention
                    .applyWithSharedKv(
                        preAttention,
                        positions,
                        {
                            sharedKvState,
                            captureKv,
                        }
                    );

            let attentionOutput = attention.output;

            if (
                this.postAttentionNorm
            ) {
                attentionOutput = this.postAttentionNorm
                        .apply(
                            attentionOutput
                        );
            }

            const attentionResidual = attentionOutput
                    .add(x);

            const preFfw = this.preFfwNorm
                    .apply(
                        attentionResidual
                    );

            let ffwOutput = this.mlp
                    .apply(
                        preFfw
                    );

            if (
                this.postFfwNorm
            ) {
                ffwOutput = this.postFfwNorm
                        .apply(
                            ffwOutput
                        );
            }

            let output = ffwOutput
                    .add(
                        attentionResidual
                    );

            if (
                this.perLayerInputDim > 0
            ) {
                const pleGate = this.perLayerInputGate
                        .apply(
                            output
                        );

                const pleActivated = this.pleActivation
                        .apply(
                            pleGate
                        );

                const pleModulated = pleActivated
                        .mul(
                            perLayerInput
                        );

                const pleProjected = this.perLayerProjection
                        .apply(
                            pleModulated
                        );

                const pleNormalized = this.postPerLayerInputNorm
                        .apply(
                            pleProjected
                        );

                output = output.add(
                        pleNormalized
                    );
            }

            const scaledOutput = output.mul(
                    this.skipScale
                );

            // TensorContainers returned from tf.tidy() survive automatically.
            // Avoid tf.keep() so the caller remains the sole lifetime owner.
            return {
                output:
                    scaledOutput,
                kvState:
                    attention.kvState,
            };
        });
    }


    applyWithCache(
        x,
        positions,
        perLayerInput,
        pastCache = null
    ) {
        return tf.tidy(() => {
            const preAttention = this.preAttentionNorm.apply(x);

            const attentionResult = this.attention.applyWithCache(
                    preAttention,
                    positions,
                    pastCache
                );

            let attentionOutput = attentionResult.output;

            if (
                this.postAttentionNorm
            ) {
                attentionOutput = this.postAttentionNorm.apply(
                        attentionOutput
                    );
            }

            const attentionResidual = attentionOutput.add(x);

            const preFfw = this.preFfwNorm.apply(
                    attentionResidual
                );

            let ffwOutput = this.mlp.apply(preFfw);

            if (this.postFfwNorm) {
                ffwOutput = this.postFfwNorm.apply(
                        ffwOutput
                    );
            }

            let output = ffwOutput.add(
                    attentionResidual
                );

            if (
                this.perLayerInputDim > 0
            ) {
                const pleGate = this.perLayerInputGate.apply(
                        output
                    );

                const pleActivated = this.pleActivation.apply(
                        pleGate
                    );

                const pleModulated = pleActivated.mul(
                        perLayerInput
                    );

                const pleProjected = this.perLayerProjection.apply(
                        pleModulated
                    );

                const pleNormalized = this.postPerLayerInputNorm.apply(
                        pleProjected
                    );

                output = output.add(
                        pleNormalized
                    );
            }

            output = output.mul(
                    this.skipScale
                );

            return {
                // Returned tensors automatically survive this tidy().
                output,
                cache:
                    attentionResult.cache
            };
        });
    }

    dispose() {
        this.preAttentionNorm.dispose();
        this.attention.dispose();
        this.postAttentionNorm?.dispose();

        this.preFfwNorm.dispose();
        this.mlp.dispose();
        this.postFfwNorm?.dispose();

        this.perLayerInputGate?.dispose();
        this.perLayerProjection?.dispose();
        this.postPerLayerInputNorm?.dispose();

        this.skipScale.dispose();
    }
}
