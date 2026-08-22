import * as tf from "@tensorflow/tfjs";

/**
 * Gemma 4 RMSNorm.
 *
 * This class stores the logical multiplicative scale directly.
 * The DeepMind JAX reference initializes scale to 1.
 *
 * If we later load a Hugging Face checkpoint whose RMSNorm parameter
 * convention is "weight" with forward scale = (1 + weight), the weight
 * loader should convert it before calling setScale().
 */
export class RMSNorm {
    constructor(dim, eps = 1e-6, withScale = true) {
        this.dim = dim;
        this.eps = eps;
        this.withScale = withScale;
        this.scale = withScale ? tf.ones([dim], "float32") : null;
    }

    setScale(scale) {
        if (!this.withScale) {
            throw new Error("This RMSNorm instance has withScale=false.");
        }
        if (scale.shape.length !== 1 || scale.shape[0] !== this.dim) {
            throw new Error(
                `RMSNorm scale shape must be [${this.dim}], got [${scale.shape}]`
            );
        }
        this.scale?.dispose();
        this.scale = scale.clone();
    }

    apply(x) {
        return tf.tidy(() => {
            const variance = tf.mean(tf.square(x), -1, true);
            let y = x.mul(tf.rsqrt(variance.add(this.eps)));

            if (this.withScale) {
                y = y.mul(this.scale);
            }

            return y;
        });
    }

    dispose() {
        this.scale?.dispose();
    }
}
