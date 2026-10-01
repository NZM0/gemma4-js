import {
    OfficialW4A16Gemma4Block
} from "./official-w4a16-layer0.js?v=0.2.2";

export function getOfficialW4A16LayerSpec(layerIndex) {
    if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= 35) {
        throw new Error(`layerIndex must be 0..34, got ${layerIndex}`);
    }

    const full = (layerIndex + 1) % 5 === 0;

    const firstKvSharedLayerIndex = 15;
    const isKvSharedLayer = layerIndex >= firstKvSharedLayerIndex;

    const storeFullLengthKv = layerIndex === 13 ||
        layerIndex === 14;

    return {
        layerIndex,
        full,
        headDim: full ? 512 : 256,
        intermediateSize: layerIndex < 15 ? 6144 : 12288,
        isKvSharedLayer,
        storeFullLengthKv,
    };
}

export function createOfficialW4A16E2BBlock(
    layerIndex,
    { outputChunkSize = 256 } = {}
) {
    const spec = getOfficialW4A16LayerSpec(layerIndex);

    return new OfficialW4A16Gemma4Block(
        {
            hiddenSize: 1536,
            intermediateSize: spec.intermediateSize,
            numAttentionHeads: 8,
            numKeyValueHeads: 1,
            headDim: spec.headDim,
            attentionType: spec.full
                ? "full_attention"
                : "sliding_attention",
            slidingWindow: 512,
            ropeTheta: spec.full
                ? 1000000.0
                : 10000.0,
            ropeProportion: spec.full
                ? 0.25
                : 1.0,
            ropeScaleFactor: 1.0,
            attnLogitSoftcap: null,
            rmsNormEps: 1e-6,
            perLayerInputDim: 256,
            usePostAttentionNorm: true,
            usePostFfwNorm: true,
            isKvSharedLayer: spec.isKvSharedLayer,
            storeFullLengthKv: spec.storeFullLengthKv,
        },
        {
            groupSize: 32,
            outputChunkSize,
        }
    );
}
