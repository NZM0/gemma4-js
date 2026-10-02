const UTF8_ENCODER = new TextEncoder();

const UTF8_DECODER = new TextDecoder(
    "utf-8", { fatal: false }
);

function asRegexSource(pattern) {
    if (typeof pattern === "string") {
        return pattern;
    }

    if (pattern && typeof pattern === "object") {
        if (typeof pattern.Regex === "string") {
            return pattern.Regex;
        }

        if (typeof pattern.String === "string") {
            return pattern.String.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        }
    }

    throw new Error(
        `Unsupported Split pattern: ${JSON.stringify(pattern)}`
    );
}

function splitChars(text) {
    return Array.from(text);
}

function byteToken(value) {
    return (
        "<0x"
        + value
            .toString(16)
            .toUpperCase()
            .padStart(2, "0")
        + ">"
    );
}

function parseByteToken(token) {
    const match = /^<0x([0-9A-Fa-f]{2})>$/.exec(token);

    return match ? Number.parseInt(match[1], 16) : null;
}

function flattenDecoder(decoder) {
    if (!decoder) {
        return [];
    }

    if (decoder.type === "Sequence") {
        return (decoder.decoders ?? []).flatMap(flattenDecoder);
    }

    return [decoder];
}

function collectPretokenizers(value) {
    if (!value) {
        return [];
    }

    if (value.type === "Sequence") {
        return (value.pretokenizers ?? value.pre_tokenizers ?? []).flatMap(collectPretokenizers);
    }

    return [value];
}

function applySplitPretokenizer(parts, config) {
    const source = asRegexSource(config.pattern);
    const regex = new RegExp(source, "gu");
    const behavior = config.behavior ?? "Isolated";
    const invert = Boolean(config.invert);
    const result = [];

    for (const part of parts) {
        if (part.special) {
            result.push(part);

            continue;
        }

        const text = part.text;
        let cursor = 0;
        const matches = Array.from(text.matchAll(regex));

        const push = (value, isMatch) => {
            if (value.length === 0) {
                return;
            }

            const selected = invert ? !isMatch : isMatch;

            if (!selected) {
                result.push({
                    text: value,
                    special: false,
                });

                return;
            }

            if (behavior === "Removed") {
                return;
            }

            if (behavior === "MergedWithPrevious" && result.length > 0
                && !result[result.length - 1].special
            ) {
                result[result.length - 1].text += value;

                return;
            }

            if (behavior === "MergedWithNext") {
                result.push({
                    text: value,
                    special: false,
                    mergeNext: true,
                });

                return;
            }

            result.push({
                text: value,
                special: false,
            });
        };

        for (const match of matches) {
            const index = match.index ?? 0;

            push(text.slice(cursor, index), false);
            push(match[0], true);

            cursor = index + match[0].length;
        }

        push(text.slice(cursor), false);
    }

    const merged = [];

    for (const part of result) {
        if (merged.length > 0 && merged[merged.length - 1].mergeNext) {
            const previous = merged.pop();

            merged.push({
                text: previous.text + part.text,
                special: false,
            });

            continue;
        }
        merged.push(part);
    }

    return merged;
}

function applyMetaspace(parts, config) {
    const replacement = config.replacement ?? "▁";
    const prependScheme = config.prepend_scheme ?? config.add_prefix_space ? "always" : "never";
    const split = config.split ?? true;
    const output = [];

    for (let index = 0; index < parts.length; index++) {
        const part = parts[index];

        if (part.special) {
            output.push(part);

            continue;
        }

        let text = part.text.replaceAll(" ", replacement);

        const shouldPrepend = prependScheme === "always" || (prependScheme === "first" && index === 0);

        if (shouldPrepend && !text.startsWith(replacement)) {
            text = replacement + text;
        }

        if (!split) {
            output.push({
                text,
                special: false,
            });

            continue;
        }

        const chunks = [];

        let start = 0;

        for (let i = 1; i < text.length; i++) {
            if (text.startsWith(replacement, i)) {
                chunks.push(text.slice(start, i));

                start = i;
            }
        }

        chunks.push(text.slice(start));

        for (const chunk of chunks) {
            if (chunk.length > 0) {
                output.push({
                    text: chunk,
                    special: false,
                });
            }
        }
    }

    return output;
}

function applyWhitespaceSplit(parts) {
    const result = [];

    for (const part of parts) {
        if (part.special) {
            result.push(part);

            continue;
        }

        for (const piece of part.text.split(/\s+/u)) {
            if (piece.length > 0) {
                result.push({
                    text: piece,
                    special: false,
                });
            }
        }
    }

    return result;
}

function normalizeText(text, normalizer) {
    if (!normalizer) {
        return text;
    }

    if (normalizer.type === "Sequence") {
        return (normalizer.normalizers ?? []).reduce((value, item) => normalizeText(value, item), text);
    }

    if (normalizer.type === "NFC" || normalizer.type === "NFD" || normalizer.type === "NFKC"
        || normalizer.type === "NFKD"
    ) {
        return text.normalize(normalizer.type);
    }

    if (normalizer.type === "Lowercase") {
        return text.toLowerCase();
    }

    if (normalizer.type === "Replace") {
        const regex = new RegExp(asRegexSource(normalizer.pattern), "gu");

        return text.replace(regex, normalizer.content ?? "");
    }

    if (normalizer.type === "Prepend") {
        return (normalizer.prepend ?? "") + text;
    }

    // Gemma tokenizer.json currently does not require a precompiled
    // SentencePiece normalizer for the text-only path used here.
    if (normalizer.type === "Precompiled") {
        throw new Error("Precompiled tokenizer normalizer is not supported by the standalone runtime.");
    }

    throw new Error(
        `Unsupported normalizer: ${normalizer.type}`
    );
}

export class Gemma4Tokenizer {
    constructor(
        tokenizerJson,
        tokenizerConfig = {},
        generationConfig = {}
    ) {
        this.json = tokenizerJson;
        this.config = tokenizerConfig;
        this.generationConfig = generationConfig;
        this.model = tokenizerJson.model;

        if (this.model?.type !== "BPE") {
            throw new Error(
                `Standalone tokenizer currently expects BPE tokenizer.json, got ${this.model?.type}.`
            );
        }

        this.vocab = new Map(Object.entries(this.model.vocab));
        this.idToToken = [];

        for (const [token, id] of this.vocab) {
            this.idToToken[Number(id)] = token;
        }

        this.mergeRanks = new Map();

        const merges = this.model.merges ?? [];

        for (let rank = 0; rank < merges.length; rank++) {
            const item = merges[rank];
            const pair = Array.isArray(item) ? item : String(item).split(" ");

            if (pair.length >= 2) {
                this.mergeRanks.set(
                    pair[0] + "\u0000" + pair[1],
                    {
                        rank,
                        result: pair[0] + pair[1],
                    }
                );
            }
        }

        this.unkToken = this.model.unk_token ?? tokenizerConfig.unk_token ?? "<unk>";
        this.unkId = Number(this.vocab.get(this.unkToken) ?? 3);
        this.byteFallback = Boolean(this.model.byte_fallback);
        this.ignoreMerges = Boolean(this.model.ignore_merges);
        this.addedTokens = (tokenizerJson.added_tokens ?? []).map(item => ({
            ...item,
            id: Number(item.id),
            })
        );

        for (const item of this.addedTokens) {
            this.idToToken[item.id] = item.content;
            this.vocab.set(item.content, item.id);
        }

        this.specialByText = new Map(
            this.addedTokens
                .filter(item => item.special)
                .map(item => [item.content, item])
        );

        this.specialTexts = Array.from(this.specialByText.keys())
            .sort((a, b) => b.length - a.length);

        this.pretokenizers = collectPretokenizers(tokenizerJson.pre_tokenizer);
        this.decoders = flattenDecoder(tokenizerJson.decoder);
        this.bosToken = tokenizerConfig.bos_token ?? "<bos>";
        this.eosToken = tokenizerConfig.eos_token ?? "<eos>";
        this.turnStartToken = tokenizerConfig.sot_token ?? "<|turn>";
        this.turnEndToken = tokenizerConfig.eot_token ?? "<turn|>";
        this.thinkToken = tokenizerConfig.think_token ?? "<|think|>";
    }

    static async fromDirectory(baseUrl) {
        const fetchJson = async (name, required = true) => {
            const response = await fetch(
                `${baseUrl}/${name}`,
                {
                    cache: "no-store",
                }
            );

            if (!response.ok) {
                if (!required) {
                    return {};
                }

                throw new Error(
                    `Failed to load ${name}: HTTP ${response.status}`
                );
            }

            return response.json();
        };

        const [tokenizerJson, tokenizerConfig, generationConfig] = await Promise.all([
            fetchJson("tokenizer.json"),

            fetchJson("tokenizer_config.json"),

            fetchJson("generation_config.json", false),
        ]);

        return new Gemma4Tokenizer(tokenizerJson, tokenizerConfig, generationConfig);
    }

    splitAddedTokens(text) {
        const result = [];
        let cursor = 0;

        while (cursor < text.length) {
            let best = null;
            let bestIndex = Infinity;

            for (const token of this.specialTexts) {
                const index = text.indexOf(token, cursor);

                if (index >= 0 && index < bestIndex) {
                    bestIndex = index;
                    best = token;
                }
            }

            if (!best) {
                result.push({
                    text: text.slice(cursor),
                    special: false,
                });

                break;
            }

            if (bestIndex > cursor) {
                result.push({
                    text: text.slice(cursor, bestIndex),
                    special: false,
                });
            }

            result.push({
                text: best,
                special: true,
            });

            cursor = bestIndex + best.length;
        }

        return result;
    }

    pretokenize(text) {
        let parts = this.splitAddedTokens(normalizeText(text, this.json.normalizer));

        for (const config of this.pretokenizers) {
            if (config.type === "Metaspace") {
                parts = applyMetaspace(parts, config);

                continue;
            }

            if (config.type === "WhitespaceSplit") {
                parts = applyWhitespaceSplit(parts);

                continue;
            }

            if (config.type === "Split") {
                parts = applySplitPretokenizer(parts, config);

                continue;
            }

            if (config.type === "ByteLevel") {
                throw new Error("ByteLevel pre-tokenizer is not expected for the Gemma 4 standalone path.");
            }

            throw new Error(
                `Unsupported pre-tokenizer: ${config.type}`
            );
        }

        return parts;
    }

    encodePiece(piece) {
        if (this.vocab.has(piece) && this.ignoreMerges) {
            return [Number(this.vocab.get(piece))];
        }

        let symbols = splitChars(piece);

        // SentencePiece BPE byte fallback: characters not directly
        // representable by the BPE alphabet fall back to UTF-8 bytes.
        const expanded = [];

        for (const symbol of symbols) {
            if (this.vocab.has(symbol)) {
                expanded.push(symbol);

                continue;
            }

            if (this.byteFallback) {
                const bytes = UTF8_ENCODER.encode(symbol);

                for (const value of bytes) {
                    expanded.push(byteToken(value));
                }

                continue;
            }

            expanded.push(symbol);
        }

        symbols = expanded;

        if (symbols.length < 2) {
            return symbols.map(symbol => Number(this.vocab.get(symbol) ?? this.unkId));
        }

        while (symbols.length > 1) {
            let bestIndex = -1;
            let bestRank = Infinity;
            let bestResult = null;

            for (let i = 0; i < symbols.length - 1; i++) {
                const merge = this.mergeRanks.get(symbols[i] + "\u0000" + symbols[i + 1]);

                if (merge && merge.rank < bestRank) {
                    bestRank = merge.rank;
                    bestIndex = i;
                    bestResult = merge.result;
                }
            }

            if (bestIndex < 0) {
                break;
            }

            const left = symbols[bestIndex];
            const right = symbols[bestIndex + 1];
            const merged = bestResult ?? (left + right);

            symbols.splice(bestIndex, 2, merged);
        }

        const ids = [];

        for (const symbol of symbols) {
            if (this.vocab.has(symbol)) {
                ids.push(Number(this.vocab.get(symbol)));

                continue;
            }

            if (this.byteFallback) {
                const bytes = UTF8_ENCODER.encode(symbol);

                for (const value of bytes) {
                    const fallback = byteToken(value);

                    ids.push(Number(this.vocab.get(fallback) ?? this.unkId));
                }
                continue;
            }
            ids.push(this.unkId);
        }

        return ids;
    }

    encode(text) {
        const ids = [];

        for (const part of this.pretokenize(text)) {
            if (part.special) {
                const token = this.specialByText.get(part.text);

                if (!token) {
                    throw new Error(
                        `Special token not found: ${part.text}`
                    );
                }
                ids.push(token.id);
                continue;
            }

            if (part.text.length > 0) {
                ids.push(...this.encodePiece(part.text));
            }
        }

        return ids;
    }

    decodeTokens(tokens) {
        let parts = tokens.slice();

        for (const decoder of this.decoders) {
            if (decoder.type === "Replace") {
                const pattern = decoder.pattern　?? {};
                const from = pattern.String ?? pattern.Regex;
                const to = decoder.content ?? "";

                if (typeof from === "string") {
                    parts = parts.map(token => token.replaceAll(from, to));
                }

                continue;
            }

            if (decoder.type === "ByteFallback") {
                const next = [];
                let bytes = [];

                const flush = () => {
                    if (bytes.length > 0) {
                        next.push(UTF8_DECODER.decode(new Uint8Array(bytes)));
                        bytes = [];
                    }
                };

                for (const token of parts) {
                    const value = parseByteToken(token);

                    if (value == null) {
                        flush();
                        next.push(token);
                    } else {
                        bytes.push(value);
                    }
                }
                flush();
                parts = next;

                continue;
            }

            if (decoder.type === "Fuse") {
                parts = [parts.join("")];

                continue;
            }

            if (decoder.type === "Strip") {
                const content = decoder.content ?? " ";
                const left = Number(decoder.start ?? decoder.left ?? 0);
                const right = Number(decoder.stop ?? decoder.right ?? 0);
                let value = parts.join("");

                for (let i = 0; i < left; i++) {
                    if (value.startsWith(content)) {
                        value = value.slice(content.length);
                    }
                }

                for (let i = 0; i < right; i++) {
                    if (value.endsWith(content)) {
                        value = value.slice(0, -content.length);
                    }
                }

                parts = [value];

                continue;
            }

            if (decoder.type === "Metaspace") {
                const replacement = decoder.replacement ?? "▁";

                let value = parts
                    .join("")
                    .replaceAll(replacement, " ");

                const prepend = decoder.prepend_scheme ?? "always";

                if (prepend !== "never" && value.startsWith(" ")) {
                    value = value.slice(1);
                }
                parts = [value];
                continue;
            }

            throw new Error(
                `Unsupported decoder: ${decoder.type}`
            );
        }

        return parts.join("");
    }

    decode(
        ids,
        {
            skipSpecialTokens = true,
        } = {}
    ) {
        const tokens = [];

        for (const rawId of ids) {
            const id = Number(rawId);
            const token = this.idToToken[id];

            if (token == null) {
                tokens.push(this.unkToken);
                continue;
            }

            const added = this.addedTokens.find(item => item.id === id);

            if (skipSpecialTokens && added?.special) {
                continue;
            }

            tokens.push(token);
        }

        return this.decodeTokens(tokens);
    }

    renderChat(
        messages,
        {
            addGenerationPrompt = true,
            enableThinking = false,
        } = {}
    ) {
        let output = this.bosToken;

        // Gemma 4's official template enables reasoning by opening a system
        // turn containing <|think|>. For the currently supported simple
        // user/assistant text-chat path, this reproduces the relevant branch
        // without requiring a Jinja runtime.
        if (enableThinking) {
            output += this.turnStartToken + "system\n" + this.thinkToken + "\n" + this.turnEndToken + "\n";
        }

        for (const message of messages) {
            if (message.role !== "user" && message.role !== "assistant") {
                throw new Error(
                    `Standalone Gemma 4 text template currently supports user/assistant messages, got ${message.role}.`
                );
            }
            output += this.turnStartToken + message.role + "\n" + String(message.content) + this.turnEndToken + "\n";
        }

        if (addGenerationPrompt) {
            output += this.turnStartToken + "model\n";
        }

        return output;
    }

    encodeChat(
        messages,
        options = {}
    ) {
        const rendered = this.renderChat(messages, options);

        return {
            rendered,
            ids: this.encode(rendered),
        };
    }

    stopTokenIds() {
        const result = new Set();

        const add = value => {
            if (value == null) {
                return;
            }

            if (Array.isArray(value)) {
                value.forEach(add);

                return;
            }

            result.add(Number(value));
        };

        const lookup = token => {
            const id = this.vocab.get(token);

            if (id != null) {
                result.add(Number(id));
            }
        };
        
        add(this.generationConfig.eos_token_id);
        lookup(this.eosToken);
        lookup(this.turnEndToken);

        return result;
    }

    describe() {
        return {
            modelType: this.model.type,
            vocabSize: this.vocab.size,
            merges: this.mergeRanks.size,
            byteFallback: this.byteFallback,
            pretokenizers: this.pretokenizers.map(item => item.type),
            decoders: this.decoders.map(item => item.type),
            specialTokens: this.specialByText.size,
        };
    }
}
