/**
 * Google Gemma 4 W4A16 compressed-tensors helpers.
 *
 * Observed checkpoint layout:
 *   weight_packed : I32 [outDim, inDim / 8]
 *   weight_scale  : BF16 [outDim, inDim / 32]
 *   weight_shape  : I64 [2]
 *
 * Config:
 *   num_bits   = 4
 *   group_size = 32
 *   strategy   = group
 *   symmetric  = true
 */
export function signedInt4FromNibble(nibble) {
    // compressed-tensors offset-binary INT4:
    // stored nibble 0..15 -> logical value -8..7
    return (nibble & 0x0f) - 8;
}

export function unpackInt4WordLE(word) {
    const result = new Int8Array(8);
    for (let i = 0; i < 8; i++) {
        const nibble = (word >>> (i * 4)) & 0x0f;
        result[i] = signedInt4FromNibble(nibble);
    }
    return result;
}


const _bf16Buffer = new ArrayBuffer(4);
const _bf16View = new DataView(_bf16Buffer);

export function roundFloat32ToBF16(value) {
    // Match the BF16 weight materialization used by the official
    // compressed-tensors dequantization reference.
    //
    // Round-to-nearest-even before truncating the lower 16 mantissa bits.
    _bf16View.setFloat32(0, value, true);
    let bits = _bf16View.getUint32(0, true);

    const lsb = (bits >>> 16) & 1;
    bits = (bits + 0x7fff + lsb) >>> 0;
    bits &= 0xffff0000;

    _bf16View.setUint32(0, bits, true);
    return _bf16View.getFloat32(0, true);
}

export function dequantizeOfficialW4A16Rows(
    packedValues,
    scales,
    {
        outDim,
        inDim,
        rowStart,
        rowCount,
        groupSize = 32,
    }
) {
    if (inDim % 8 !== 0) {
        throw new Error("inDim must be divisible by 8.");
    }
    if (inDim % groupSize !== 0) {
        throw new Error("inDim must be divisible by groupSize.");
    }

    const packedInDim = inDim / 8;
    const groupsPerRow = inDim / groupSize;

    if (packedValues.length !== outDim * packedInDim) {
        throw new Error("packedValues length mismatch.");
    }
    if (scales.length !== outDim * groupsPerRow) {
        throw new Error("scale length mismatch.");
    }

    const result = new Float32Array(rowCount * inDim);

    for (let localRow = 0; localRow < rowCount; localRow++) {
        const row = rowStart + localRow;
        const packedRowOffset = row * packedInDim;
        const scaleRowOffset = row * groupsPerRow;
        const outputRowOffset = localRow * inDim;

        for (let packedCol = 0; packedCol < packedInDim; packedCol++) {
            const word = packedValues[packedRowOffset + packedCol];
            const baseInput = packedCol * 8;

            for (let k = 0; k < 8; k++) {
                const inputIndex = baseInput + k;
                const nibble = (word >>> (k * 4)) & 0x0f;
                const q = signedInt4FromNibble(nibble);
                const group = Math.floor(inputIndex / groupSize);
                const scale = scales[scaleRowOffset + group];

                result[outputRowOffset + inputIndex] = q * scale;
            }
        }
    }

    return result;
}

export function officialW4A16StorageBreakdown({
    outDim,
    inDim,
    groupSize = 32,
    scaleBytes = 2,
}) {
    const parameterCount = outDim * inDim;
    const packedBytes = parameterCount / 2;
    const scaleCount = outDim * (inDim / groupSize);
    const scalesBytes = scaleCount * scaleBytes;
    const totalBytes = packedBytes + scalesBytes;

    return {
        parameterCount,
        packedBytes,
        scaleCount,
        scaleBytes: scalesBytes,
        totalBytes,
        bitsPerParameter: totalBytes * 8 / parameterCount,
    };
}
