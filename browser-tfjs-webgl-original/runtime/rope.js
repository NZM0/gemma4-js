const tf = globalThis.tf;

/**
 * Port of google-deepmind/gemma:
 * gemma/gm/math/_positional_embeddings.py::apply_rope
 *
 * inputs:    [B, L, N, H]
 * positions: [B, L]
 */
export function applyRoPE(
    inputs,
    positions,
    {
        baseFrequency = 10000,
        scaleFactor = 1.0,
        ropeProportion = 1.0,
    } = {}
) {
    if (scaleFactor < 1.0) {
        throw new Error(`scaleFactor must be >= 1.0, got ${scaleFactor}`);
    }

    return tf.tidy(() => {
        const headDim = inputs.shape.at(-1);

        if (headDim % 2 !== 0) {
            throw new Error(`headDim must be even, got ${headDim}`);
        }

        const halfDim = headDim / 2;
        const ropeAngles = Math.floor((ropeProportion * headDim) / 2);
        const nopeAngles = halfDim - ropeAngles;

        // freq_exponents = (2 / head_dim) * arange(rope_angles)
        const exponents = tf
            .range(0, ropeAngles, 1, "float32")
            .mul(2.0 / headDim);

        let timescaleRotary = tf.pow(
            tf.scalar(baseFrequency, "float32"),
            exponents
        );

        // Non-RoPE dimensions are represented by timescale = inf,
        // so position / inf = 0 => sin=0, cos=1.
        let timescale;
        if (nopeAngles > 0) {
            const inf = tf.fill([nopeAngles], Infinity, "float32");
            timescale = tf.concat([timescaleRotary, inf], 0);
        } else {
            timescale = timescaleRotary;
        }

        // [B, L, halfDim]
        let sinusoid = positions
            .toFloat()
            .expandDims(-1)
            .div(timescale.reshape([1, 1, halfDim]));

        sinusoid = sinusoid.div(scaleFactor);

        // [B, L, 1, halfDim]
        const sin = tf.sin(sinusoid).expandDims(2);
        const cos = tf.cos(sinusoid).expandDims(2);

        const [firstHalf, secondHalf] = tf.split(inputs, 2, -1);

        const firstPart = firstHalf.mul(cos).sub(secondHalf.mul(sin));
        const secondPart = secondHalf.mul(cos).add(firstHalf.mul(sin));

        return tf.concat([firstPart, secondPart], -1);
    });
}
