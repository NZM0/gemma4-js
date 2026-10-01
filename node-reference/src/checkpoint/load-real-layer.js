import { NodeSafeTensorsReader } from "./node-safetensors-reader.js";

export function getRealLayerSpec(layerIndex) {
    if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= 35) {
        throw new Error(`layerIndex must be 0..34, got ${layerIndex}`);
    }

    const full = (layerIndex + 1) % 5 === 0;
    const headDim = full ? 512 : 256;
    const intermediateSize = layerIndex < 15 ? 6144 : 12288;

    // Gemma 4 E2B has 20 vertically KV-shared layers.
    // Layer 13 seeds sliding-attention KV and layer 14 seeds
    // full-attention KV. Layers 15..34 reuse those states.
    const isKvSharedLayer = layerIndex >= 15;

    const captureSharedKv = layerIndex === 13 ||
        layerIndex === 14;

    return {
        full,
        headDim,
        qOut: 8 * headDim,
        intermediateSize,
        isKvSharedLayer,
        captureSharedKv,
        attentionType:
            full
                ? "full_attention"
                : "sliding_attention",
        root: `model.language_model.layers.${layerIndex}`,
    };
}

export function realLayerTensorNames(layerIndex) {
    const { root } = getRealLayerSpec(layerIndex);

    return {
        preAttentionNorm: `${root}.input_layernorm.weight`,
        skipScale: `${root}.layer_scalar`,
        mlpDown: `${root}.mlp.down_proj.weight`,
        mlpGate: `${root}.mlp.gate_proj.weight`,
        mlpUp: `${root}.mlp.up_proj.weight`,
        pleGate: `${root}.per_layer_input_gate.weight`,
        pleProjection: `${root}.per_layer_projection.weight`,
        postAttentionNorm: `${root}.post_attention_layernorm.weight`,
        postFfwNorm: `${root}.post_feedforward_layernorm.weight`,
        postPleNorm: `${root}.post_per_layer_input_norm.weight`,
        preFfwNorm: `${root}.pre_feedforward_layernorm.weight`,
        kNorm: `${root}.self_attn.k_norm.weight`,
        kProj: `${root}.self_attn.k_proj.weight`,
        oProj: `${root}.self_attn.o_proj.weight`,
        qNorm: `${root}.self_attn.q_norm.weight`,
        qProj: `${root}.self_attn.q_proj.weight`,
        vProj: `${root}.self_attn.v_proj.weight`,
    };
}

function assertShape(actual, expected, name) {
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
        throw new Error(
            `${name}: expected shape [${expected}], got [${actual}]`
        );
    }
}

async function setNorm(reader, norm, name, expectedShape) {
    const t = await reader.readTensor(name);
    assertShape(t.shape, expectedShape, name);
    norm.setScale(t);
    t.dispose();
}

async function setLinear(reader, linear, name, expectedShape) {
    const t = await reader.readTensor(name);
    assertShape(t.shape, expectedShape, name);

    const transposed = t.transpose();
    linear.setWeight(transposed);

    t.dispose();
    transposed.dispose();
}

export async function loadRealGemma4Layer(
    block,
    layerIndex,
    {
        safetensorsPath,
        reader = null,
    }
) {
    const ownsReader = reader === null;
    const actualReader = reader ?? new NodeSafeTensorsReader(
        safetensorsPath
    );

    const s = getRealLayerSpec(layerIndex);
    const n = realLayerTensorNames(layerIndex);

    try {
        await setNorm(actualReader, block.preAttentionNorm, n.preAttentionNorm, [1536]);
        await setNorm(actualReader, block.postAttentionNorm, n.postAttentionNorm, [1536]);
        await setNorm(actualReader, block.preFfwNorm, n.preFfwNorm, [1536]);
        await setNorm(actualReader, block.postFfwNorm, n.postFfwNorm, [1536]);
        await setNorm(actualReader, block.postPerLayerInputNorm, n.postPleNorm, [1536]);

        await setNorm(actualReader, block.attention.qNorm, n.qNorm, [s.headDim]);

        await setLinear(actualReader, block.attention.qProj, n.qProj, [s.qOut, 1536]);
        await setLinear(actualReader, block.attention.oProj, n.oProj, [1536, s.qOut]);

        if (
            !s.isKvSharedLayer
        ) {
            await setNorm(actualReader, block.attention.kNorm, n.kNorm, [s.headDim]);
            await setLinear(actualReader, block.attention.kProj, n.kProj, [s.headDim, 1536]);
            await setLinear(actualReader, block.attention.vProj, n.vProj, [s.headDim, 1536]);
        }

        await setLinear(actualReader, block.mlp.gateProj, n.mlpGate, [s.intermediateSize, 1536]);
        await setLinear(actualReader, block.mlp.upProj, n.mlpUp, [s.intermediateSize, 1536]);
        await setLinear(actualReader, block.mlp.downProj, n.mlpDown, [1536, s.intermediateSize]);

        await setLinear(actualReader, block.perLayerInputGate, n.pleGate, [256, 1536]);
        await setLinear(actualReader, block.perLayerProjection, n.pleProjection, [1536, 256]);

        const scalar = await actualReader.readTensor(n.skipScale);
        assertShape(scalar.shape, [1], n.skipScale);
        block.setSkipScale(scalar);
        scalar.dispose();
    } finally {
        if (ownsReader) {
            await actualReader.close();
        }
    }
}
