import { Gemma4Block } from "./block.js";
import { getRealLayerSpec } from "./checkpoint/load-real-layer.js";

export function createRealE2BBlock(layerIndex) {
    const s = getRealLayerSpec(layerIndex);

    return new Gemma4Block({
        hiddenSize: 1536,
        intermediateSize: s.intermediateSize,

        numAttentionHeads: 8,
        numKeyValueHeads: 1,
        headDim: s.headDim,

        attentionType: s.full ? "full_attention" : "sliding_attention",

        slidingWindow: 512,

        ropeTheta: s.full ? 1000000.0 : 10000.0,

        ropeProportion: s.full ? 0.25 : 1.0,

        ropeScaleFactor: 1.0,
        attnLogitSoftcap: null,
        rmsNormEps: 1e-6,

        perLayerInputDim: 256,
        usePostAttentionNorm: true,
        usePostFfwNorm: true,
    });
}
