import * as tf from "@tensorflow/tfjs";
import { Linear } from "./linear.js";

/**
 * PyTorch-style tanh approximation of GELU:
 *
 * GELU(x) ~= 0.5 * x *
 *   (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 *
 * Gemma 4 E2B config uses:
 * hidden_activation = "gelu_pytorch_tanh"
 *
 * TensorFlow.js does not expose tf.gelu() at the top level,
 * so we implement the activation directly from the formula.
 */
export class GELUPytorchTanh {
    apply(x) {
        return tf.tidy(() => {
            const xCubed = tf.pow(x, 3);

            const inner = x
                .add(xCubed.mul(0.044715))
                .mul(Math.sqrt(2 / Math.PI));

            return x
                .mul(0.5)
                .mul(tf.tanh(inner).add(1.0));
        });
    }
}

/**
 * Gemma 4 dense gated MLP:
 *
 * gate = gate_proj(x)
 * up   = up_proj(x)
 *
 * y = GELU_PYTORCH_TANH(gate) * up
 * out = down_proj(y)
 *
 * Step 4 will compare this output against the reference implementation
 * after loading the official checkpoint weights.
 */
export class Gemma4MLP {
    constructor(hiddenSize, intermediateSize) {
        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;

        this.gateProj = new Linear(
            hiddenSize,
            intermediateSize
        );

        this.upProj = new Linear(
            hiddenSize,
            intermediateSize
        );

        this.downProj = new Linear(
            intermediateSize,
            hiddenSize
        );

        this.activation = new GELUPytorchTanh();
    }

    apply(x) {
        return tf.tidy(() => {
            const gate = this.gateProj.apply(x);
            const up = this.upProj.apply(x);

            const activatedGate =
                this.activation.apply(gate);

            const hidden =
                activatedGate.mul(up);

            return this.downProj.apply(hidden);
        });
    }

    dispose() {
        this.gateProj.dispose();
        this.upProj.dispose();
        this.downProj.dispose();
    }
}
