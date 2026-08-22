import * as tf from "@tensorflow/tfjs";

export class Linear {
    constructor(inDim, outDim, { useBias = false } = {}) {
        this.inDim = inDim;
        this.outDim = outDim;
        this.useBias = useBias;

        const std = 0.02;
        this.weight = tf.randomNormal([inDim, outDim], 0, std, "float32");
        this.bias = useBias ? tf.zeros([outDim], "float32") : null;
    }

    setWeight(weight) {
        if (
            weight.shape.length !== 2 ||
            weight.shape[0] !== this.inDim ||
            weight.shape[1] !== this.outDim
        ) {
            throw new Error(
                `Linear weight must be [${this.inDim}, ${this.outDim}], got [${weight.shape}]`
            );
        }

        this.weight.dispose();
        this.weight = weight.clone();
    }

    apply(x) {
        return tf.tidy(() => {
            let y = tf.matMul(x, this.weight);
            if (this.bias) {
                y = y.add(this.bias);
            }
            return y;
        });
    }

    dispose() {
        this.weight.dispose();
        this.bias?.dispose();
    }
}
