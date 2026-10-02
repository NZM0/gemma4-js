import {
    Gemma4Tokenizer
} from "./runtime/standalone-tokenizer.js?v=0.3.0";

import {
    TfjsGemma4Backend,
    TFJS_WEBGL_BUILD
} from "./runtime/tfjs-reference-engine.js?v=0.3.0";

const $ = selector => document.querySelector(selector);
const output = $("#output");
const response = $("#response");
const tokenizerStatus = $("#tokenizer-status");
const loadBtn = $("#load");
const generateBtn = $("#generate");

let tokenizer = null;
let model = null;

function log(...items) {
    output.textContent += items.join(" ") + "\n";

    output.scrollTop = output.scrollHeight;
}

async function loadTokenizer() {
    if (tokenizer) {
        return tokenizer;
    }

    tokenizer = await Gemma4Tokenizer.fromDirectory("../tokenizer/gemma4");

    tokenizerStatus.textContent = "ready";

    return tokenizer;
}

loadBtn.onclick = async () => {
    output.textContent = "";

    const modelFile = $("#model-file").files[0];

    if (!modelFile) {
        log("ERROR: select model.safetensors.");

        return;
    }

    loadBtn.disabled = true;

    generateBtn.disabled = true;

    try {
        log("TF.js WebGL build:", TFJS_WEBGL_BUILD);
        await loadTokenizer();

        model?.dispose();

        model = new TfjsGemma4Backend(
            modelFile,
            {
                backend: "webgl",
                log,
            }
        );

        const started = performance.now();

        await model.load();

        log(
            `load complete: ${((performance.now() - started) / 1000).toFixed(2)} s`
        );

        log("tf.memory():", JSON.stringify(model.memorySummary()));

        generateBtn.disabled = false;
    } catch (error) {
        console.error(error);

        log("ERROR:", error.stack ?? error.message ?? String(error));

        model?.dispose();

        model = null;
    } finally {
        loadBtn.disabled = false;
    }
};

generateBtn.onclick = async () => {
    if (!model || !tokenizer) {
        return;
    }

    generateBtn.disabled = true;

    response.textContent = "";

    model.resetCache();

    const prompt = $("#prompt").value;
    const maxNewTokens = Number($("#max-tokens").value);
    const enableThinking = $("#thinking").value === "true";

    const {
        rendered,
        ids: inputIds,
    } = tokenizer.encodeChat(
        [
            {
                role: "user",
                content: prompt,
            }
        ],
        {
            enableThinking,
            addGenerationPrompt: true,
        }
    );

    const stopIds = tokenizer.stopTokenIds();

    log("");
    log("========================================");
    log("GENERATION");
    log("========================================");
    log("prompt:", JSON.stringify(prompt));
    log("rendered:", JSON.stringify(rendered));
    log("input ids:", JSON.stringify(inputIds));

    const generated = [];

    try {
        let result = await model.greedy(inputIds, 0);

        for (let step = 0; step < maxNewTokens; step++) {
            const tokenId = result.nextTokenId;

            generated.push(tokenId);

            response.textContent = tokenizer.decode(
                generated,
                {
                    skipSpecialTokens: true,
                }
            );

            log(
                `token ${step + 1}:`,
                tokenId,
                JSON.stringify(
                    tokenizer.decode(
                        [tokenId],
                        {
                            skipSpecialTokens: false,
                        }
                    )
                )
            );

            if (stopIds.has(tokenId)) {
                log("stop token reached.");

                break;
            }

            if (step + 1 >= maxNewTokens) {
                break;
            }

            result = await model.greedy([tokenId], inputIds.length + step);
        }

        log("cache:", JSON.stringify(model.cacheSummary()));
    } catch (error) {
        console.error(error);

        log("ERROR:", error.stack ?? error.message ?? String(error));
    } finally {
        generateBtn.disabled = false;
    }
};
