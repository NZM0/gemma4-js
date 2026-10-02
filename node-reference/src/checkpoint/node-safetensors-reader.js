import fs from "node:fs/promises";
import * as tf from "@tensorflow/tfjs";

function product(shape) {
    return shape.reduce((a, b) => a * b, 1);
}

export function bf16BufferToFloat32(buffer) {
    if (buffer.byteLength % 2 !== 0) {
        throw new Error(
            `BF16 byte length must be even, got ${buffer.byteLength}`
        );
    }

    const count = buffer.byteLength / 2;

    // Copy into a tightly packed ArrayBuffer so Uint16Array alignment
    // and byteOffset are predictable.
    const copy = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    const u16 = new Uint16Array(copy);
    const u32 = new Uint32Array(count);

    for (let i = 0; i < count; i++) {
        u32[i] = u16[i] << 16;
    }

    return new Float32Array(u32.buffer);
}

export class NodeSafeTensorsReader {
    constructor(filePath) {
        this.filePath = filePath;
        this.headerLength = null;
        this.dataBaseOffset = null;
        this.header = null;
        this.metadata = {};
        this.byName = new Map();
        this.handle = null;
    }

    async open() {
        if (this.handle) {
            return this;
        }

        this.handle = await fs.open(this.filePath, "r");

        const prefix = Buffer.allocUnsafe(8);
        const prefixRead = await this.handle.read(prefix, 0, 8, 0);

        if (prefixRead.bytesRead !== 8) {
            throw new Error("Invalid SafeTensors file: could not read 8-byte header length.");
        }

        this.headerLength = Number(prefix.readBigUInt64LE(0));

        if (!Number.isSafeInteger(this.headerLength) || this.headerLength <= 0) {
            throw new Error(
                `Invalid SafeTensors header length: ${this.headerLength}`
            );
        }

        const headerBuffer = Buffer.allocUnsafe(this.headerLength);
        const headerRead = await this.handle.read(headerBuffer, 0, this.headerLength, 8);

        if (headerRead.bytesRead !== this.headerLength) {
            throw new Error(
                `Short SafeTensors header read: ${headerRead.bytesRead}/${this.headerLength}`
            );
        }

        this.header = JSON.parse(headerBuffer.toString("utf8"));

        this.metadata = this.header.__metadata__
            ?? {};

        this.byName = new Map(
            Object.entries(this.header)
                .filter(([name]) => name !== "__metadata__")
        );

        this.dataBaseOffset = 8 + this.headerLength;

        return this;
    }

    getMetadata(name) {
        const meta = this.byName.get(name);

        if (!meta) {
            throw new Error(
                `Tensor not found in header: ${name}`
            );
        }

        return meta;
    }

    async readRaw(name) {
        await this.open();

        const meta = this.getMetadata(name);
        const offsets = meta.data_offsets ?? meta.dataOffsets;
        const [relativeStart, relativeEnd] = offsets;
        const length = relativeEnd - relativeStart;
        const buffer = Buffer.allocUnsafe(length);
        const absoluteStart = this.dataBaseOffset + relativeStart;

        const {
            bytesRead
        } = await this.handle.read(buffer, 0, length, absoluteStart);

        if (bytesRead !== length) {
            throw new Error(
                `Short read for ${name}: ` + `${bytesRead}/${length}`
            );
        }

        return {
            buffer,
            meta
        };
    }

    async readFloat32(name) {
        const {
            buffer,
            meta
        } = await this.readRaw(name);

        if (meta.dtype !== "BF16") {
            throw new Error(
                `${name}: expected BF16, got ${meta.dtype}`
            );
        }

        const values = bf16BufferToFloat32(buffer);
        const expected = product(meta.shape);

        if (values.length !== expected) {
            throw new Error(
                `${name}: decoded ${values.length} values, ` + `expected ${expected} from shape [${meta.shape}]`
            );
        }

        return {
            values,
            shape: meta.shape
        };
    }

    async readTensor(name) {
        const {
            values,
            shape
        } = await this.readFloat32(name);

        return tf.tensor(values, shape, "float32");
    }

    /**
     * Read selected rows from a 2-D BF16 tensor without loading the full table.
     *
     * SafeTensors payload is contiguous row-major storage.
     * Each requested row is range-read independently.
     */
    async readRowsFloat32(name, rowIndices) {
        await this.open();

        const meta = this.getMetadata(name);

        if (meta.dtype !== "BF16" || meta.shape.length !== 2) {
            throw new Error(
                `${name}: readRowsFloat32 requires a 2-D BF16 tensor`
            );
        }

        const [numRows, rowWidth] = meta.shape;
        const offsets = meta.data_offsets ?? meta.dataOffsets;
        const rowByteLength = rowWidth * 2;
        const values = new Float32Array(rowIndices.length * rowWidth);

        for (let i = 0; i < rowIndices.length; i++) {
            const rowIndex = rowIndices[i];

            if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= numRows) {
                throw new Error(
                    `${name}: invalid row index ${rowIndex}`
                );
            }

            const relativeStart = offsets[0] + rowIndex * rowByteLength;
            const buffer = Buffer.allocUnsafe(rowByteLength);
            const absoluteStart = this.dataBaseOffset + relativeStart;

            const {
                bytesRead
            } = await this.handle.read(buffer, 0, rowByteLength, absoluteStart);

            if (bytesRead !== rowByteLength) {
                throw new Error(
                    `${name}: short row read for index ${rowIndex}`
                );
            }

            const row = bf16BufferToFloat32(buffer);

            values.set(row, i * rowWidth);
        }

        return {
            values,
            shape: [rowIndices.length, rowWidth]
        };
    }

    async readRowsTensor(name, rowIndices) {
        const {
            values,
            shape
        } = await this.readRowsFloat32(name, rowIndices);

        return tf.tensor2d(values, shape, "float32");
    }

    /**
     * Efficiently read one contiguous [startRow, startRow + rowCount)
     * range from a 2-D BF16 tensor with a single fs.read().
     */
    async readRowRangeFloat32(name, startRow, rowCount) {
        await this.open();

        const meta = this.getMetadata(name);

        if (meta.dtype !== "BF16" || meta.shape.length !== 2) {
            throw new Error(
                `${name}: readRowRangeFloat32 requires a 2-D BF16 tensor`
            );
        }

        const [numRows, rowWidth] = meta.shape;

        if (!Number.isInteger(startRow) || !Number.isInteger(rowCount) || startRow < 0 || rowCount <= 0 || startRow + rowCount > numRows
        ) {
            throw new Error(
                `${name}: invalid row range start=${startRow} count=${rowCount}`
            );
        }

        const offsets = meta.data_offsets ?? meta.dataOffsets;
        const rowBytes = rowWidth * 2;
        const byteLength = rowCount * rowBytes;
        const relativeStart = offsets[0] + startRow * rowBytes;
        const absoluteStart = this.dataBaseOffset + relativeStart;
        const buffer = Buffer.allocUnsafe(byteLength);

        const {
            bytesRead
        } = await this.handle.read(buffer, 0, byteLength, absoluteStart);

        if (bytesRead !== byteLength) {
            throw new Error(
                `${name}: short contiguous row read ${bytesRead}/${byteLength}`
            );
        }

        return {
            values: bf16BufferToFloat32(buffer),

            shape: [rowCount, rowWidth]
        };
    }

    async readRowRangeTensor(name, startRow, rowCount) {
        const {
            values,
            shape
        } = await this.readRowRangeFloat32(name, startRow, rowCount);

        return tf.tensor2d(values, shape, "float32");
    }

    async close() {
        await this.handle?.close();
        this.handle = null;
    }
}
