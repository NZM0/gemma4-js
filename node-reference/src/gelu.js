import * as tf from "@tensorflow/tfjs";

/**
 * PyTorch-compatible GELU tanh approximation:
 *
 * 0.5 * x * (1 + tanh(
 *     sqrt(2/pi) * (x + 0.044715 * x^3)
 * ))
 */
export function geluTanh(x) {
    return tf.tidy(() => {
        const c =
            Math.sqrt(
                2 /
                Math.PI
            );

        const x3 =
            x.mul(x)
                .mul(x);

        const inner =
            x.add(
                x3.mul(
                    0.044715
                )
            )
            .mul(c);

        return x
            .mul(0.5)
            .mul(
                tf.tanh(inner)
                    .add(1)
            );
    });
}
