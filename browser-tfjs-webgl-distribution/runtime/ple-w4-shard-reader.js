
function bf16BitsToFloat32(bits) {
    const buffer = new ArrayBuffer(4);
    const view = new DataView(buffer);
    view.setUint32(0, (bits << 16) >>> 0, true);
    return view.getFloat32(0, true);
}

export class LocalPleW4ShardReader {
    constructor(manifest, shardFiles) {
        this.manifest = manifest;
        this.width = Number(manifest.width);
        this.vocabSize = Number(manifest.vocab_size);
        this.rowsPerShard = Number(manifest.layout.rows_per_shard);
        this.packedRowBytes = Number(manifest.layout.packed_row_bytes);
        this.scaleRowBytes = Number(manifest.layout.scale_row_bytes);
        this.shards = manifest.shards;
        this.fileByName = new Map(
            Array.from(shardFiles).map(file => [file.name, file])
        );

        const missing = this.shards
                .map(x => x.file)
                .filter(name => !this.fileByName.has(name));

        if (missing.length) {
            throw new Error(
                `PLE shard directory is incomplete: ${missing.length} missing; first=${missing[0]}`
            );
        }
    }

    static async fromFiles(manifestFile, shardFiles) {
        const manifest = JSON.parse(
                await manifestFile.text()
            );

        if (
            manifest.format !== "gemma4-ple-w4-sharded-v1" ||
            manifest.complete !== true
        ) {
            throw new Error(
                `Unexpected/incomplete PLE manifest: ${manifest.format}`
            );
        }

        if (
            Number(manifest.width) !== 8960 ||
            Number(manifest.quantization?.group_size) !== 32
        ) {
            throw new Error(
                "Unexpected PLE W4 geometry."
            );
        }

        return new LocalPleW4ShardReader(
            manifest,
            shardFiles
        );
    }

    _location(tokenId) {
        const id = Number(tokenId);

        if (
            !Number.isInteger(id) ||
            id < 0 ||
            id >= this.vocabSize
        ) {
            throw new Error(
                `PLE token out of range: ${tokenId}`
            );
        }

        const shardIndex = Math.floor(
                id /
                this.rowsPerShard
            );

        const shard = this.shards[
                shardIndex
            ];

        return {
            shard,
            localRow:
                id -
                Number(
                    shard.row_start
                ),
        };
    }

    async readRows(ids) {
        const output = new Float32Array(
                ids.length *
                this.width
            );

        await Promise.all(
            ids.map(
                async (
                    tokenId,
                    outputRow
                ) => {
                    const {
                        shard,
                        localRow,
                    } = this._location(
                            tokenId
                        );

                    const file = this.fileByName.get(
                            shard.file
                        );

                    const packedStart = Number(
                            shard.packed_offset
                        ) +
                        localRow *
                        this.packedRowBytes;

                    const scaleStart = Number(
                            shard.scales_offset
                        ) +
                        localRow *
                        this.scaleRowBytes;

                    const [
                        packedBuffer,
                        scaleBuffer,
                    ] = await Promise.all([
                            file
                                .slice(
                                    packedStart,
                                    packedStart +
                                    this.packedRowBytes
                                )
                                .arrayBuffer(),

                            file
                                .slice(
                                    scaleStart,
                                    scaleStart +
                                    this.scaleRowBytes
                                )
                                .arrayBuffer(),
                        ]);

                    const packed = new Uint8Array(
                            packedBuffer
                        );

                    const scaleBits = new Uint16Array(
                            scaleBuffer
                        );

                    const base = outputRow *
                        this.width;

                    for (
                        let i = 0;
                        i < this.width;
                        ++i
                    ) {
                        const byte = packed[
                                i >> 1
                            ];

                        const nibble = (i & 1) === 0
                                ? byte & 0x0f
                                : byte >> 4;

                        const q = nibble >= 8
                                ? nibble - 16
                                : nibble;

                        const scale = bf16BitsToFloat32(
                                scaleBits[
                                    Math.floor(
                                        i /
                                        32
                                    )
                                ]
                            );

                        output[
                            base +
                            i
                        ] = q *
                            scale;
                    }
                }
            )
        );

        return {
            values:
                output,
            shape: [
                ids.length,
                this.width,
            ],
        };
    }
}
