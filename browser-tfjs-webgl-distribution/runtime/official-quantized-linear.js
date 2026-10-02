const tf = globalThis.tf;

import {
    dequantizeOfficialW4A16Rows,
    officialW4A16StorageBreakdown
} from "./official-w4a16.js";

export class OfficialW4A16Linear {
    constructor(
        inDim, outDim,
        {
            groupSize = 32,
            outputChunkSize = 256,
        } = {}
    ) {
        this.inDim = inDim;
        this.outDim = outDim;
        this.groupSize = groupSize;
        this.outputChunkSize = outputChunkSize;

        this.denseWeight?.dispose();
        this.denseWeight = null;
        this.packedWeight = null;
        this.scales = null;
        this.storageMode = null;
        this.denseWeight = null;
        this.storageMode = null;
    }

    setOfficialWeights({
        packedWeight, scales,
    }) {
        if (!(packedWeight instanceof Int32Array)) {
            throw new Error("packedWeight must be Int32Array.");
        }
        if (!(scales instanceof Float32Array)) {
            throw new Error("scales must be Float32Array.");
        }

        const packedExpected = this.outDim * (this.inDim / 8);
        const scaleExpected = this.outDim * (this.inDim / this.groupSize);

        if (packedWeight.length !== packedExpected) {
            throw new Error(
                `packed weight length ${packedWeight.length} != ${packedExpected}`
            );
        }

        if (scales.length !== scaleExpected) {
            throw new Error(
                `scale length ${scales.length} != ${scaleExpected}`
            );
        }

        this.denseWeight?.dispose();
        this.denseWeight = null;
        this.packedWeight = packedWeight;
        this.scales = scales;
        this.storageMode = "w4a16";
    }

    setDenseWeight(weight) {
        if (weight.shape.length !== 2) {
            throw new Error(
                `Dense weight must be rank-2, got [${weight.shape}]`
            );
        }

        if (weight.shape[0] !== this.outDim || weight.shape[1] !== this.inDim) {
            throw new Error(
                `Dense weight shape [${weight.shape}] != ` + `[${this.outDim}, ${this.inDim}]`
            );
        }

        this.denseWeight?.dispose();
        this.denseWeight = weight.clone();
        this.packedWeight = null;
        this.scales = null;
        this.storageMode = "dense";
    }

    apply(x) {
        if (!this.storageMode) {
            throw new Error("Official checkpoint weight is not loaded.");
        }

        if (x.shape[x.shape.length - 1] !== this.inDim) {
            throw new Error(
                `Expected input last dim ${this.inDim}, got [${x.shape}]`
            );
        }

        if (this.storageMode === "dense") {
            return tf.tidy(() => tf.matMul(x, this.denseWeight, false, true));
        }

        const outputs = [];

        try {
            for (let rowStart = 0; rowStart < this.outDim; rowStart += this.outputChunkSize) {
                const rowCount = Math.min(this.outputChunkSize, this.outDim - rowStart);

                const restored = dequantizeOfficialW4A16Rows(
                    this.packedWeight, this.scales,
                    {
                        outDim: this.outDim,
                        inDim: this.inDim,
                        rowStart, rowCount,
                        groupSize: this.groupSize,
                    }
                );

                const yChunk = tf.tidy(() => {
                    const w = tf.tensor2d(restored, [rowCount, this.inDim], "float32");

                    return tf.matMul(x, w, false, true);
                });

                outputs.push(yChunk);
            }

            return outputs.length === 1 ? outputs[0].clone() : tf.concat(outputs, -1);
        } finally {
            for (const tensor of outputs) {
                tensor.dispose();
            }
        }
    }

    storageBreakdown() {
        return officialW4A16StorageBreakdown({
            outDim: this.outDim,
            inDim: this.inDim,
            groupSize: this.groupSize,
            scaleBytes: 2,
        });
    }

    dispose() {
        this.denseWeight?.dispose?.();
        this.denseWeight = null;
        this.packedWeight = null;
        this.scales = null;
        this.storageMode = null;
    }
}
