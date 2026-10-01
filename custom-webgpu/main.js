import {
    CorePlusRemoteW4PleReader,
    CorePlusW4PleShardReader
} from "./gpu-runtime.js?v=github-release-1.0.0-rc3";

import {
    Gemma4WebGPU
} from "./gemma4-webgpu.js?v=github-release-1.0.0-rc3";

import {
    Gemma4Tokenizer
} from "./standalone-tokenizer.js?v=0.1.1";

const $ = selector =>
        document.querySelector(
            selector
        );

const output = $("#output");

const responseEl = $("#response");

const reasoningEl = $("#reasoning");

const reasoningWrapEl = $("#reasoning-wrap");

const statusEl = $("#status");

const prefillEl = $("#prefill");

const decodeEl = $("#decode");

const tpsEl = $("#tps");

const tokenizerStatusEl = $("#tokenizer-status");

const tokenCounterEl = $("#token-counter");

const loadBtn = $("#load");

const generateBtn = $("#generate");

const stopBtn = $("#stop");

let model = null;

let tokenizer = null;

let reader = null;

function syncSourceUi() {
    const mode = $("#model-source")
            ?.value
        ??
        "local";

    const local = $("#local-source-fields");

    const remote = $("#remote-source-fields");

    if (local) {
        local.hidden = mode !==
            "local";
    }

    if (remote) {
        remote.hidden = mode !==
            "remote";
    }
}

let stopRequested = false;

function log(
    ...items
) {
    output.textContent +=
        items.join(
            " "
        )
        +
        "\n";

    output.scrollTop = output.scrollHeight;
}

function setStatus(
    value
) {
    statusEl.textContent = value;
}

function fmtBytes(
    n
) {
    if (
        n >=
        1024 ** 3
    ) {
        return (
            n /
            1024 ** 3
        ).toFixed(
            2
        )
        +
        " GiB";
    }

    if (
        n >=
        1024 ** 2
    ) {
        return (
            n /
            1024 ** 2
        ).toFixed(
            2
        )
        +
        " MiB";
    }

    return (
        n /
        1024
    ).toFixed(
        1
    )
    +
    " KiB";
}

function estimate(
    header
) {
    let q = 0;

    for (
        const [
            name,
            tensor
        ]
        of
        Object.entries(
            header
        )
    ) {
        if (
            (
                name.includes(
                    "model.language_model.layers."
                )
                ||
                name.startsWith(
                    "model.language_model.per_layer_model_projection."
                )
            )
            &&
            (
                name.endsWith(
                    ".weight_packed"
                )
                ||
                name.endsWith(
                    ".weight_scale"
                )
            )
        ) {
            q +=
                tensor.data_offsets[1]
                -
                tensor.data_offsets[0];
        }
    }

    const emb = header[
            "model.language_model.embed_tokens.weight"
        ];

    const lm = emb
            ? emb.data_offsets[1]
              -
              emb.data_offsets[0]
            : 0;

    return {
        q,
        lm,
        total:
            q + lm,
    };
}

async function requestGPU() {
    if (
        !navigator.gpu
    ) {
        throw new Error(
            "WebGPU unavailable. Open http://localhost, not http://[::]."
        );
    }

    const adapter = await navigator.gpu
            .requestAdapter({
                powerPreference:
                    "high-performance",
            });

    if (!adapter) {
        throw new Error(
            "No WebGPU adapter."
        );
    }

    const device = await adapter
            .requestDevice();

    device.lost.then(
        info =>
            log(
                "DEVICE LOST:",
                info.message
            )
    );

    return device;
}

async function loadTokenizer() {
    tokenizerStatusEl.textContent = "LOADING";

    tokenizer = await Gemma4Tokenizer
            .fromDirectory(
                "../tokenizer/gemma4"
            );

    tokenizerStatusEl.textContent = "READY";

    log(
        "Standalone tokenizer:",
        JSON.stringify(
            tokenizer.describe()
        )
    );
}


function splitReasoningOutput(
    rawText,
    plainText
) {
    const thoughtStart = "<|channel>thought\n";

    const channelEnd = "<channel|>";

    const thoughtIndex = rawText.indexOf(
            thoughtStart
        );

    if (
        thoughtIndex <
        0
    ) {
        return {
            reasoning:
                "",
            answer:
                plainText,
        };
    }

    const bodyStart = thoughtIndex
        +
        thoughtStart.length;

    const end = rawText.indexOf(
            channelEnd,
            bodyStart
        );

    if (
        end <
        0
    ) {
        return {
            reasoning:
                rawText.slice(
                    bodyStart
                ),
            answer:
                "",
        };
    }

    const reasoning = rawText.slice(
            bodyStart,
            end
        );

    const tailIdsText = rawText.slice(
            end
            +
            channelEnd.length
        );

    const cleanedAnswer = tailIdsText
            .replace(
                /^<\|channel>final\n/u,
                ""
            )
            .replace(
                /<channel\|>$/u,
                ""
            )
            .trimStart();

    return {
        reasoning,
        answer:
            cleanedAnswer
            ||
            plainText,
    };
}

$("#model-source").onchange = syncSourceUi;

syncSourceUi();

$("#clear").onclick = () => {
        output.textContent = "";
    };

loadBtn.onclick = async () => {
        loadBtn.disabled = true;

        generateBtn.disabled = true;

        setStatus(
            "LOADING"
        );

        try {
            if (!tokenizer) {
                await loadTokenizer();
            }

            const sourceMode = $("#model-source")
                    .value;

            if (
                sourceMode ===
                "remote"
            ) {
                const baseUrl = $("#remote-base-url")
                        .value
                        .trim();

                if (
                    !baseUrl ||
                    baseUrl.includes(
                        "<"
                    )
                ) {
                    throw new Error(
                        "Set the remote model base URL (for example a Hugging Face resolve/main URL)."
                    );
                }

                reader = await CorePlusRemoteW4PleReader
                        .fromBaseUrl(
                            baseUrl
                        );
            } else {
                const coreFile = $("#core-file")
                        .files[0];

                const pleFiles = Array.from(
                        $("#ple-shard-directory")
                            .files
                        ??
                        []
                    );

                if (!coreFile) {
                    throw new Error(
                        "Select core.safetensors."
                    );
                }

                if (
                    pleFiles.length === 0
                ) {
                    throw new Error(
                        "Select the webgpu-distribution/ple-w4 directory."
                    );
                }

                const manifestCandidates = pleFiles.filter(
                        file =>
                            file.name ===
                            "manifest.json"
                    );

                if (
                    manifestCandidates.length === 0
                ) {
                    throw new Error(
                        "PLE directory does not contain manifest.json. Select webgpu-distribution/ple-w4/."
                    );
                }

                let manifestFile = null;

                for (
                    const candidate of
                    manifestCandidates
                ) {
                    try {
                        const manifest = JSON.parse(
                                await candidate.text()
                            );

                        if (
                            manifest.format ===
                            "gemma4-ple-w4-sharded-v1"
                            &&
                            manifest.complete ===
                            true
                        ) {
                            manifestFile = candidate;

                            break;
                        }
                    } catch {
                        // Keep searching if an unrelated JSON file is present.
                    }
                }

                if (!manifestFile) {
                    throw new Error(
                        "No complete Gemma 4 PLE W4 manifest was found in the selected directory."
                    );
                }

                const shardFiles = pleFiles.filter(
                        file =>
                            file.name.endsWith(
                                ".bin"
                            )
                    );

                reader = await CorePlusW4PleShardReader
                        .fromFiles(
                            coreFile,
                            manifestFile,
                            shardFiles
                        );
            }

            const size = estimate(
                    reader.header
                );

            log(
                ""
            );

            log(
                "========================================"
            );

            log(
                "CUSTOM WEBGPU / LOCAL + REMOTE DISTRIBUTION"
            );

            log(
                "build          : github-release-1.0.0-rc3"
            );

            log(
                "========================================"
            );

            log(
                ""
            );

            log(
                "[runtime dependencies]"
            );

            log(
                "Transformers.js: NONE"
            );

            log(
                "ONNX Runtime   : NONE"
            );

            log(
                "Web ML runtime : NONE"
            );

            log(
                "Tokenizer      : standalone JS"
            );

            log(
                "Model backend  : custom WebGPU/WGSL"
            );

            log(
                ""
            );

            log(
                "[model sources]"
            );

            log(
                "source mode    :",
                $("#model-source").value
            );

            if (
                $("#model-source").value ===
                "remote"
            ) {
                log(
                    "CORE           : remote core.safetensors / cached range chunks"
                );

                log(
                    "PLE            : remote W4 shards / Browser Cache"
                );
            } else {
                log(
                    "CORE           : local core.safetensors"
                );

                log(
                    "PLE            : local full signed W4 shard set"
                );
            }

            log(
                ""
            );

            log(
                "[resident GPU weight estimate]"
            );

            log(
                "W4A16 linears:",
                fmtBytes(
                    size.q
                )
            );

            log(
                "tied LM head :",
                fmtBytes(
                    size.lm
                )
            );

            log(
                "total        :",
                fmtBytes(
                    size.total
                )
            );

            const device = await requestGPU();

            log(
                ""
            );

            log(
                "[WebGPU limits]"
            );

            log(
                "maxBufferSize:",
                fmtBytes(
                    Number(
                        device
                            .limits
                            .maxBufferSize
                    )
                )
            );

            model = new Gemma4WebGPU(
                    device,
                    reader,
                    {
                        maxSeq:
                            Number(
                                $("#max-seq")
                                    .value
                            ),

                        lmChunkRows:
                            Number(
                                $("#lm-chunk")
                                    .value
                            ),

                        log,
                    }
                );

            const t0 = performance.now();

            await model.load();

            await device.queue
                .onSubmittedWorkDone();

            log(
                ""
            );

            log(
                "preload time:",
                (
                    (
                        performance.now()
                        -
                        t0
                    )
                    /
                    1000
                ).toFixed(
                    2
                ),
                "s"
            );

            setStatus(
                "READY"
            );

            generateBtn.disabled = false;
        } catch (
            error
        ) {
            console.error(
                error
            );

            log(
                "ERROR:",
                error.stack
                ??
                error.message
                ??
                String(
                    error
                )
            );

            setStatus(
                "ERROR"
            );

            model?.dispose();

            model = null;
        } finally {
            loadBtn.disabled = false;
        }
    };

generateBtn.onclick = async () => {
        if (
            !model
            ||
            !tokenizer
        ) {
            return;
        }

        const prompt = $("#prompt")
                .value
                .trim();

        if (!prompt) {
            return;
        }

        const maxNewTokens = Number(
                $("#max-new-tokens")
                    .value
            );

        const maxSeq = Number(
                $("#max-seq")
                    .value
            );

        stopRequested = false;

        generateBtn.disabled = true;

        stopBtn.disabled = false;

        setStatus(
            "GENERATING"
        );

        responseEl.textContent = "";

        reasoningEl.textContent = "";

        reasoningWrapEl.hidden = true;

        tokenCounterEl.textContent = "0 tokens";

        model.resetCache();

        try {
            const {
                rendered,
                ids: inputIds,
            } = tokenizer.encodeChat(
                    [
                        {
                            role:
                                "user",
                            content:
                                prompt,
                        }
                    ],
                    {
                        addGenerationPrompt:
                            true,
                        enableThinking:
                            $("#enable-thinking")
                                .checked,
                    }
                );

            log(
                ""
            );

            log(
                "========================================"
            );

            log(
                "GENERATION"
            );

            log(
                "========================================"
            );

            log(
                "prompt:",
                JSON.stringify(
                    prompt
                )
            );

            log(
                "rendered:",
                JSON.stringify(
                    rendered
                )
            );

            log(
                "input ids:",
                JSON.stringify(
                    inputIds
                )
            );

            if (
                inputIds.length >=
                maxSeq
            ) {
                throw new Error(
                    `Prompt length ${inputIds.length} exceeds KV-cache capacity ${maxSeq}.`
                );
            }

            const stopIds = tokenizer
                    .stopTokenIds();

            log(
                "stop ids:",
                JSON.stringify(
                    Array.from(
                        stopIds
                    )
                )
            );

            const generated = [];

            const decodeTimes = [];

            let t0 = performance.now();

            let result = await model.greedy(
                    inputIds,
                    0
                );

            const prefillMs = performance.now()
                -
                t0;

            prefillEl.textContent = `${prefillMs.toFixed(1)} ms`;

            let nextId = result.nextTokenId;

            log(
                `[prefill] ${inputIds.length} tokens`
            );

            log(
                "  next token:",
                nextId
            );

            log(
                "  time      :",
                `${prefillMs.toFixed(2)} ms`
            );

            log(
                "  cache     :",
                JSON.stringify(
                    model.cacheSummary()
                )
            );

            for (
                let step = 0;
                step < maxNewTokens;
                step++
            ) {
                if (
                    stopRequested
                ) {
                    log(
                        "Stopped by user."
                    );

                    break;
                }

                if (
                    stopIds.has(
                        nextId
                    )
                ) {
                    log(
                        `Stop token reached: ${nextId}`
                    );

                    break;
                }

                generated.push(
                    nextId
                );

                const text = tokenizer.decode(
                        generated,
                        {
                            skipSpecialTokens:
                                true,
                        }
                    );

                const rawText = tokenizer.decode(
                        generated,
                        {
                            skipSpecialTokens:
                                false,
                        }
                    );

                const separated = splitReasoningOutput(
                        rawText,
                        text
                    );

                reasoningWrapEl.hidden = separated.reasoning.length ===
                    0;

                reasoningEl.textContent = separated.reasoning;

                responseEl.textContent = separated.answer;

                tokenCounterEl.textContent = `${generated.length} tokens`;

                await new Promise(
                    resolve =>
                        requestAnimationFrame(
                            resolve
                        )
                );

                if (
                    generated.length >=
                    maxNewTokens
                ) {
                    break;
                }

                const absolutePosition = inputIds.length
                    +
                    generated.length
                    -
                    1;

                if (
                    absolutePosition >=
                    maxSeq
                ) {
                    log(
                        "KV-cache capacity reached."
                    );

                    break;
                }

                t0 = performance.now();

                result = await model.greedy(
                        [
                            nextId
                        ],
                        absolutePosition
                    );

                const decodeMs = performance.now()
                    -
                    t0;

                decodeTimes.push(
                    decodeMs
                );

                nextId = result.nextTokenId;

                const avg = decodeTimes
                        .reduce(
                            (
                                a,
                                b
                            ) =>
                                a + b,
                            0
                        )
                    /
                    decodeTimes.length;

                decodeEl.textContent = `${avg.toFixed(1)} ms`;

                tpsEl.textContent = (
                        1000 /
                        avg
                    ).toFixed(
                        1
                    );
            }

            const finalText = tokenizer.decode(
                    generated,
                    {
                        skipSpecialTokens:
                            true,
                    }
                );

            log(
                ""
            );

            log(
                "========================================"
            );

            log(
                "RESULT"
            );

            log(
                "========================================"
            );

            log(
                "generated ids:",
                JSON.stringify(
                    generated
                )
            );

            log(
                "text:",
                JSON.stringify(
                    finalText
                )
            );

            if (
                decodeTimes.length >
                0
            ) {
                const avg = decodeTimes
                        .reduce(
                            (
                                a,
                                b
                            ) =>
                                a + b,
                            0
                        )
                    /
                    decodeTimes.length;

                log(
                    "avg decode:",
                    `${avg.toFixed(2)} ms/token`
                );

                log(
                    "throughput:",
                    `${(1000 / avg).toFixed(2)} tokens/s`
                );
            }

            setStatus(
                stopRequested
                    ? "STOPPED"
                    : "DONE"
            );
        } catch (
            error
        ) {
            console.error(
                error
            );

            log(
                "ERROR:",
                error.stack
                ??
                error.message
                ??
                String(
                    error
                )
            );

            setStatus(
                "ERROR"
            );
        } finally {
            generateBtn.disabled = false;

            stopBtn.disabled = true;
        }
    };

stopBtn.onclick = () => {
        stopRequested = true;

        setStatus(
            "STOPPING"
        );
    };

try {
    await loadTokenizer();
} catch (
    error
) {
    console.error(
        error
    );

    tokenizerStatusEl.textContent = "ERROR";

    log(
        "Standalone tokenizer load failed:",
        error.message
    );

    log(
        "Prepare it with:"
    );

    log(
        "python python/prepare_step10_20_standalone.py <model-directory>"
    );
}
