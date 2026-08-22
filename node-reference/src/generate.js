import * as tf from "@tensorflow/tfjs";

import {
    NodeSafeTensorsReader
} from "./checkpoint/node-safetensors-reader.js";

import {
    Gemma4Tokenizer
} from "./tokenizer.js";

import {
    prepareRealInputAndPle,
    runRealTransformerStack,
    applyRealFinalNorm,
    selectNextTokenFromLastPosition
} from "./dense-runtime.js";

function usage() {
    console.log(`
Gemma 4 Node.js TensorFlow.js Reference

Usage:
  npm run generate:node -- \
    /path/to/model.safetensors \
    "こんにちは。あなたは誰？" \
    /path/to/tokenizer-directory \
    [max-new-tokens] \
    [auto|chat|base]

Examples:
  npm run generate:node -- \
    ./gemma-4-E2B-it/model.safetensors \
    "Explain quantum computing in one sentence." \
    ./gemma-4-E2B-it \
    2 \
    chat

  npm run generate:node -- \
    ./gemma-4-E2B/model.safetensors \
    "The future of AI is" \
    ./gemma-4-E2B \
    2 \
    base
`);
}

function normalizeTokenIds(value) {
    if (value == null) {
        throw new Error(
            "Tokenizer/chat template returned no token IDs."
        );
    }

    if (
        typeof value.tolist ===
        "function"
    ) {
        value =
            value.tolist();
    }

    if (
        value?.input_ids != null
    ) {
        value =
            value.input_ids;

        if (
            typeof value.tolist ===
            "function"
        ) {
            value =
                value.tolist();
        }
    }

    if (
        ArrayBuffer.isView(value)
    ) {
        return Array.from(
            value,
            Number
        );
    }

    if (!Array.isArray(value)) {
        throw new Error(
            `Unsupported token-id result: ${typeof value}`
        );
    }

    while (
        Array.isArray(value) &&
        value.length === 1 &&
        (
            Array.isArray(value[0]) ||
            ArrayBuffer.isView(value[0])
        )
    ) {
        value =
            Array.from(
                value[0]
            );
    }

    return value.map(Number);
}

async function buildPromptTokenIds(
    tokenizer,
    prompt,
    mode,
    tokenizerSource
) {
    let resolvedMode =
        mode;

    if (resolvedMode === "auto") {
        resolvedMode =
            /(?:^|[-_/])it(?:$|[-_/])/i.test(
                tokenizerSource
            )
                ? "chat"
                : "base";
    }

    if (resolvedMode === "chat") {
        const {
            rendered,
            ids,
        } =
            tokenizer.encodeChat(
                [
                    {
                        role: "user",
                        content: prompt,
                    }
                ],
                {
                    addGenerationPrompt: true,
                    enableThinking: false,
                }
            );

        return {
            tokenIds: ids,
            rendered,
            mode: "standalone Gemma 4 chat template",
        };
    }

    if (resolvedMode === "base") {
        const rendered =
            tokenizer.bosToken +
            prompt;

        return {
            tokenIds:
                tokenizer.encode(
                    rendered
                ),
            rendered,
            mode: "base prompt + BOS",
        };
    }

    throw new Error(
        `Unknown prompt mode: ${resolvedMode}`
    );
}

async function realForwardLastToken(
    reader,
    safetensorsPath,
    tokenIds,
    {
        topK = 1,
        temperature = 0.0,
    } = {}
) {
    const positions =
        tf.tensor2d(
            [
                Array.from(
                    {
                        length:
                            tokenIds.length
                    },
                    (_, i) => i
                )
            ],
            [
                1,
                tokenIds.length
            ],
            "int32"
        );

    let inputHidden = null;
    let perLayerInputs = null;
    let transformerHidden = null;
    let finalHidden = null;

    try {
        ({
            hidden:
                inputHidden,
            perLayerInputs,
        } =
            await prepareRealInputAndPle(
                reader,
                tokenIds
            ));

        transformerHidden =
            await runRealTransformerStack(
                reader,
                safetensorsPath,
                null,
                inputHidden,
                perLayerInputs,
                positions
            );

        finalHidden =
            await applyRealFinalNorm(
                reader,
                transformerHidden
            );

        return await selectNextTokenFromLastPosition(
            reader,
            finalHidden,
            {
                chunkRows: 2048,
                topK,
                temperature,
            }
        );
    } finally {
        inputHidden?.dispose();
        perLayerInputs?.dispose();
        transformerHidden?.dispose();
        finalHidden?.dispose();
        positions.dispose();
    }
}

const [
    ,
    ,
    safetensorsPath,
    prompt,
    tokenizerSource,
    maxNewTokensArg =
        "4",
    promptMode =
        "auto",
] = process.argv;

if (
    !safetensorsPath
    ||
    prompt == null
    ||
    !tokenizerSource
) {
    usage();
    process.exit(2);
}

const maxNewTokens =
    Number.parseInt(
        maxNewTokensArg,
        10
    );

if (
    !Number.isInteger(
        maxNewTokens
    ) ||
    maxNewTokens <= 0
) {
    throw new Error(
        `Invalid max-new-tokens: ${maxNewTokensArg}`
    );
}

await tf.setBackend("cpu");
await tf.ready();

console.log(
    "========================================"
);

console.log(
    "Gemma 4 — Node.js TensorFlow.js Generation"
);

console.log(
    "========================================"
);

console.log(
    "backend         :",
    tf.getBackend()
);

console.log(
    "tokenizer       :",
    tokenizerSource
);

console.log(
    "max new tokens  :",
    maxNewTokens
);

console.log(
    "\nLoading tokenizer..."
);

const tokenizer =
    await Gemma4Tokenizer
        .fromDirectory(
            tokenizerSource
        );

const {
    tokenIds: promptTokenIds,
    mode: resolvedPromptMode,
    rendered: renderedPrompt,
} =
    await buildPromptTokenIds(
        tokenizer,
        prompt,
        promptMode,
        tokenizerSource
    );

console.log(
    "prompt mode     :",
    resolvedPromptMode
);

console.log(
    "prompt          :",
    prompt
);

console.log(
    "rendered        :",
    JSON.stringify(
        renderedPrompt
    )
);

console.log(
    "prompt tokens   :",
    promptTokenIds.length
);

console.log(
    "prompt token ids:",
    promptTokenIds
);

const reader =
    new NodeSafeTensorsReader(
        safetensorsPath
    );

await reader.open();

const generated =
    [...promptTokenIds];

const newTokenIds = [];

console.log(
    "\n[GENERATION]"
);

console.log(
    "Reference mode: dense BF16 checkpoint, layer-streamed TensorFlow.js CPU."
);

console.log(
    "The full prefix is recomputed for each generated token to keep memory bounded.\n"
);

try {
    for (
        let step = 0;
        step < maxNewTokens;
        step++
    ) {
        const started =
            Date.now();

        console.log(
            `token ${step + 1}/${maxNewTokens}: running ${generated.length}-token prefix...`
        );

        const next =
            await realForwardLastToken(
                reader,
                safetensorsPath,
                generated,
                {
                    topK: 1,
                    temperature: 0.0,
                }
            );

        generated.push(
            next.tokenId
        );

        newTokenIds.push(
            next.tokenId
        );

        const piece =
            await tokenizer.decode(
                [next.tokenId],
                {
                    skipSpecialTokens:
                        false
                }
            );

        const partial =
            await tokenizer.decode(
                newTokenIds,
                {
                    skipSpecialTokens:
                        false
                }
            );

        const seconds =
            (
                (
                    Date.now() -
                    started
                ) /
                1000
            ).toFixed(2);

        console.log(
            `  id=${next.tokenId} logit=${next.logit.toFixed(5)} time=${seconds}s`
        );

        console.log(
            `  piece  : ${JSON.stringify(piece)}`
        );

        console.log(
            `  output : ${partial}`
        );

        if (
            tokenizer
                .stopTokenIds()
                .has(
                    next.tokenId
                )
        ) {
            console.log(
                "  stop token reached."
            );

            break;
        }
    }
} finally {
    await reader.close();
}

const rawText =
    await tokenizer.decode(
        newTokenIds,
        {
            skipSpecialTokens:
                false
        }
    );

const cleanText =
    await tokenizer.decode(
        newTokenIds,
        {
            skipSpecialTokens:
                true
        }
    );

console.log(
    "\n========================================"
);

console.log(
    "RESULT"
);

console.log(
    "========================================"
);

console.log(
    "raw  :",
    rawText
);

console.log(
    "clean:",
    cleanText
);

console.log(
    "new token ids:",
    newTokenIds
);

const memory =
    tf.memory();

console.log(
    "\nfinal tf.memory():",
    memory
);

if (
    memory.numTensors !== 0 ||
    memory.numDataBuffers !== 0 ||
    memory.numBytes !== 0
) {
    throw new Error(
        `Tensor leak detected: ${memory.numTensors} tensors, ${memory.numBytes} bytes`
    );
}

console.log(
    "\nGeneration completed."
);
