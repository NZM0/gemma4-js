const tf = globalThis.tf;

import {
    getOfficialW4A16LayerSpec
} from "../official-w4a16-layer-factory.js";

function hasTensor(reader, name) {
    return Object.prototype.hasOwnProperty.call(
        reader.header,
        name
    );
}

async function loadLinear(
    reader,
    base,
    linear
) {
    const packedName = `${base}.weight_packed`;

    const denseName = `${base}.weight`;

    if (hasTensor(reader, packedName)) {
        const shape = await reader.readInt64(
                `${base}.weight_shape`
            );

        const packed = await reader.readInt32(
                packedName
            );

        const scale = await reader.readBF16Float32(
                `${base}.weight_scale`
            );

        const [
            outDim,
            inDim
        ] = shape.values;

        if (
            outDim !== linear.outDim ||
            inDim !== linear.inDim
        ) {
            throw new Error(
                `${base}: packed checkpoint ` +
                `[${outDim}, ${inDim}] != ` +
                `Linear [${linear.outDim}, ${linear.inDim}]`
            );
        }

        linear.setOfficialWeights({
            packedWeight:
                packed.values,
            scales:
                scale.values,
        });

        return "W4A16";
    }

    if (hasTensor(reader, denseName)) {
        const dense = await reader.readBF16Float32(
                denseName
            );

        if (
            dense.shape.length !== 2 ||
            dense.shape[0] !== linear.outDim ||
            dense.shape[1] !== linear.inDim
        ) {
            throw new Error(
                `${base}: dense checkpoint ` +
                `[${dense.shape}] != ` +
                `Linear [${linear.outDim}, ${linear.inDim}]`
            );
        }

        const tensor = tf.tensor2d(
                dense.values,
                dense.shape,
                "float32"
            );

        linear.setDenseWeight(
            tensor
        );

        tensor.dispose();

        return "BF16";
    }

    throw new Error(
        `${base}: neither weight_packed nor BF16 weight exists`
    );
}

async function setNormIfPresent(
    reader,
    norm,
    name,
    expectedLength
) {
    if (!hasTensor(reader, name)) {
        // RMSNorm object starts with scale = ones, which is exactly the
        // scale-free RMSNorm behavior required when no learned scale is
        // stored in the checkpoint.
        return false;
    }

    const {
        values,
        shape
    } = await reader.readBF16Float32(
            name
        );

    if (
        shape.length !== 1 ||
        shape[0] !== expectedLength
    ) {
        throw new Error(
            `${name}: expected [${expectedLength}], got [${shape}]`
        );
    }

    const t = tf.tensor1d(
            values,
            "float32"
        );

    norm.setScale(t);
    t.dispose();

    return true;
}

export async function loadOfficialW4A16Layer(
    block,
    layerIndex,
    reader
) {
    const spec = getOfficialW4A16LayerSpec(
            layerIndex
        );

    const root = `model.language_model.layers.${layerIndex}`;

    await setNormIfPresent(
        reader,
        block.preAttentionNorm,
        `${root}.input_layernorm.weight`,
        1536
    );

    await setNormIfPresent(
        reader,
        block.postAttentionNorm,
        `${root}.post_attention_layernorm.weight`,
        1536
    );

    await setNormIfPresent(
        reader,
        block.preFfwNorm,
        `${root}.pre_feedforward_layernorm.weight`,
        1536
    );

    await setNormIfPresent(
        reader,
        block.postFfwNorm,
        `${root}.post_feedforward_layernorm.weight`,
        1536
    );

    await setNormIfPresent(
        reader,
        block.postPerLayerInputNorm,
        `${root}.post_per_layer_input_norm.weight`,
        1536
    );

    const qNormStored = await setNormIfPresent(
            reader,
            block.attention.qNorm,
            `${root}.self_attn.q_norm.weight`,
            spec.headDim
        );

    let kNormStored = false;

    if (!spec.isKvSharedLayer) {
        kNormStored = await setNormIfPresent(
                reader,
                block.attention.kNorm,
                `${root}.self_attn.k_norm.weight`,
                spec.headDim
            );
    }

    const modes = {};

    modes.q = await loadLinear(
            reader,
            `${root}.self_attn.q_proj`,
            block.attention.qProj
        );

    if (!spec.isKvSharedLayer) {
        modes.k = await loadLinear(
                reader,
                `${root}.self_attn.k_proj`,
                block.attention.kProj
            );

        modes.v = await loadLinear(
                reader,
                `${root}.self_attn.v_proj`,
                block.attention.vProj
            );
    } else {
        modes.k = "SHARED";
        modes.v = "SHARED";
    }

    modes.o = await loadLinear(
            reader,
            `${root}.self_attn.o_proj`,
            block.attention.oProj
        );

    modes.gate = await loadLinear(
            reader,
            `${root}.mlp.gate_proj`,
            block.mlp.gateProj
        );

    modes.up = await loadLinear(
            reader,
            `${root}.mlp.up_proj`,
            block.mlp.upProj
        );

    modes.down = await loadLinear(
            reader,
            `${root}.mlp.down_proj`,
            block.mlp.downProj
        );

    modes.pleGate = await loadLinear(
            reader,
            `${root}.per_layer_input_gate`,
            block.perLayerInputGate
        );

    modes.pleProjection = await loadLinear(
            reader,
            `${root}.per_layer_projection`,
            block.perLayerProjection
        );

    const scalar = await reader.readBF16Float32(
            `${root}.layer_scalar`
        );

    block.setSkipScale(
        scalar.values[0]
    );

    return {
        qNormStored,
        kNormStored,
        modes,
    };
}
