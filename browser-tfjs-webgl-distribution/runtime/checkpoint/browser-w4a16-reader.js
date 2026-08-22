function bf16ToFloat32(u16) {
    const buffer = new ArrayBuffer(4);
    const view = new DataView(buffer);
    view.setUint32(0, u16 << 16, true);
    return view.getFloat32(0, true);
}

export class BrowserW4A16Reader {
    constructor(file) {
        this.file = file;
        this.header = null;
        this.metadata = {};
        this.dataStart = null;
    }

    async open() {
        const prefix =
            await this.file
                .slice(0, 8)
                .arrayBuffer();

        const headerLength =
            Number(
                new DataView(prefix)
                    .getBigUint64(
                        0,
                        true
                    )
            );

        const headerBuffer =
            await this.file
                .slice(
                    8,
                    8 + headerLength
                )
                .arrayBuffer();

        this.header =
            JSON.parse(
                new TextDecoder(
                    "utf-8"
                )
                    .decode(
                        headerBuffer
                    )
            );

        this.metadata =
            this.header.__metadata__
            ??
            {};

        this.dataStart =
            8 +
            headerLength;

        return this;
    }

    async close() {
        // Browser File objects do not need explicit close.
    }

    info(name) {
        const info =
            this.header[name];

        if (!info) {
            throw new Error(
                `Tensor not found: ${name}`
            );
        }

        return info;
    }

    async readBytes(name) {
        const info =
            this.info(name);

        const [
            start,
            end
        ] =
            info.data_offsets;

        const buffer =
            await this.file
                .slice(
                    this.dataStart + start,
                    this.dataStart + end
                )
                .arrayBuffer();

        return {
            buffer,
            info,
        };
    }

    async readInt32(name) {
        const {
            buffer,
            info,
        } =
            await this.readBytes(
                name
            );

        if (
            info.dtype !==
            "I32"
        ) {
            throw new Error(
                `${name}: expected I32, got ${info.dtype}`
            );
        }

        return {
            values:
                new Int32Array(
                    buffer
                ),
            shape:
                info.shape,
        };
    }

    async readInt64(name) {
        const {
            buffer,
            info,
        } =
            await this.readBytes(
                name
            );

        if (
            info.dtype !==
            "I64"
        ) {
            throw new Error(
                `${name}: expected I64, got ${info.dtype}`
            );
        }

        const view =
            new DataView(
                buffer
            );

        const values =
            [];

        for (
            let offset = 0;
            offset < buffer.byteLength;
            offset += 8
        ) {
            values.push(
                Number(
                    view.getBigInt64(
                        offset,
                        true
                    )
                )
            );
        }

        return {
            values,
            shape:
                info.shape,
        };
    }

    async readBF16Float32(name) {
        const {
            buffer,
            info,
        } =
            await this.readBytes(
                name
            );

        if (
            info.dtype !==
            "BF16"
        ) {
            throw new Error(
                `${name}: expected BF16, got ${info.dtype}`
            );
        }

        return {
            values:
                this.decodeBF16(
                    buffer
                ),
            shape:
                info.shape,
        };
    }

    decodeBF16(buffer) {
        const view =
            new DataView(
                buffer
            );

        const values =
            new Float32Array(
                buffer.byteLength /
                2
            );

        for (
            let i = 0;
            i < values.length;
            i++
        ) {
            values[i] =
                bf16ToFloat32(
                    view.getUint16(
                        i * 2,
                        true
                    )
                );
        }

        return values;
    }

    async readBF16Rows(
        name,
        ids
    ) {
        const info =
            this.info(
                name
            );

        if (
            info.dtype !==
            "BF16"
            ||
            info.shape.length !==
            2
        ) {
            throw new Error(
                `${name}: readBF16Rows expects BF16 rank-2 tensor.`
            );
        }

        const width =
            info.shape[1];

        const rowBytes =
            width * 2;

        const tensorStart =
            info.data_offsets[0];

        const out =
            new Float32Array(
                ids.length *
                width
            );

        for (
            let rowIndex = 0;
            rowIndex < ids.length;
            rowIndex++
        ) {
            const id =
                Number(
                    ids[
                        rowIndex
                    ]
                );

            const begin =
                this.dataStart
                +
                tensorStart
                +
                id *
                rowBytes;

            const buffer =
                await this.file
                    .slice(
                        begin,
                        begin + rowBytes
                    )
                    .arrayBuffer();

            out.set(
                this.decodeBF16(
                    buffer
                ),
                rowIndex *
                width
            );
        }

        return {
            values:
                out,
            shape: [
                ids.length,
                width,
            ],
        };
    }

    async readBF16RowRange(
        name,
        rowStart,
        rowCount
    ) {
        const info =
            this.info(
                name
            );

        if (
            info.dtype !==
            "BF16"
            ||
            info.shape.length !==
            2
        ) {
            throw new Error(
                `${name}: readBF16RowRange expects BF16 rank-2 tensor.`
            );
        }

        const width =
            info.shape[1];

        const rowBytes =
            width * 2;

        const begin =
            this.dataStart
            +
            info.data_offsets[0]
            +
            rowStart *
            rowBytes;

        const buffer =
            await this.file
                .slice(
                    begin,
                    begin
                    +
                    rowCount *
                    rowBytes
                )
                .arrayBuffer();

        return {
            values:
                this.decodeBF16(
                    buffer
                ),
            shape: [
                rowCount,
                width,
            ],
        };
    }
}
