import * as tf from "@tensorflow/tfjs";

/**
 * General causal mask for possibly different query/key lengths.
 *
 * queryPositions: [B, T]
 * keyPositions:   [B, S]
 *
 * Returns [B, T, S].
 */
export function createPositionCausalMask(queryPositions, keyPositions) {
    return tf.tidy(() => {
        const q = queryPositions.expandDims(-1);
        const k = keyPositions.expandDims(1);

        return k.lessEqual(q);
    });
}

/**
 * Gemma local-window condition:
 *
 * key_pos > query_pos - window
 * key_pos < query_pos + window
 *
 * Causality is applied separately.
 */
export function createSlidingPositionMask(queryPositions, keyPositions, slidingWindowSize) {
    return tf.tidy(() => {
        const q = queryPositions.expandDims(-1);
        const k = keyPositions.expandDims(1);
        const lower = k.greater(q.sub(slidingWindowSize));
        const upper = k.less(q.add(slidingWindowSize));

        return tf.logicalAnd(lower, upper);
    });
}

// Backward-compatible Step 2 helpers.
export function createCausalMask(batchSize, seqLen) {
    return tf.tidy(() => {
        const positions = tf
            .range(0, seqLen, 1, "int32")
            .reshape([1, seqLen])
            .tile([batchSize, 1]);

        return createPositionCausalMask(positions, positions);
    });
}

export function createSlidingMask(positions, slidingWindowSize) {
    return createSlidingPositionMask(positions, positions, slidingWindowSize);
}
