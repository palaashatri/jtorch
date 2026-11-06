package org.jtorch.onnx.protobuf;

import java.nio.charset.StandardCharsets;
import java.util.List;

/** Minimal protobuf wire-format cursor for ONNX import (varint/length-delimited/32/64-bit). */
public final class ProtoReader {
    private final byte[] data;
    private int pos;
    private final int end;

    public ProtoReader(byte[] data) {
        this(data, 0, data.length);
    }

    public ProtoReader(byte[] data, int offset, int length) {
        this.data = data;
        this.pos = offset;
        this.end = offset + length;
    }

    public boolean exhausted() {
        return pos >= end;
    }

    /** One wire field header. */
    public record Field(int number, int wireType) {}

    public Field nextField() {
        long tag = readVarint();
        return new Field((int) (tag >>> 3), (int) (tag & 7));
    }

    public long readVarint() {
        long result = 0;
        int shift = 0;
        while (true) {
            if (pos >= end) {
                throw new IllegalArgumentException("Truncated varint in ONNX protobuf");
            }
            int b = data[pos++] & 0xFF;
            result |= (long) (b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                return result;
            }
            shift += 7;
            if (shift >= 64) {
                throw new IllegalArgumentException("Varint too large in ONNX protobuf");
            }
        }
    }

    public int readInt32() {
        return (int) readVarint();
    }

    public long readInt64() {
        return readVarint();
    }

    public double readDouble() {
        long bits = readFixed64();
        return Double.longBitsToDouble(bits);
    }

    public float readFloat() {
        int bits = readFixed32();
        return Float.intBitsToFloat(bits);
    }

    public String readString() {
        byte[] bytes = readBytes();
        return new String(bytes, StandardCharsets.UTF_8);
    }

    public byte[] readBytes() {
        int length = (int) readVarint();
        if (length < 0 || pos + length > end) {
            throw new IllegalArgumentException("Truncated length-delimited field in ONNX protobuf");
        }
        byte[] out = new byte[length];
        System.arraycopy(data, pos, out, 0, length);
        pos += length;
        return out;
    }

    public List<Long> readPackedInt64() {
        int length = (int) readVarint();
        if (pos + length > end) {
            throw new IllegalArgumentException("Truncated packed field in ONNX protobuf");
        }
        int fieldEnd = pos + length;
        java.util.List<Long> out = new java.util.ArrayList<>();
        while (pos < fieldEnd) {
            out.add(readVarint());
        }
        if (pos != fieldEnd) {
            throw new IllegalArgumentException("Packed field overrun in ONNX protobuf");
        }
        return List.copyOf(out);
    }

    public List<Float> readPackedFloat() {
        int length = (int) readVarint();
        if (pos + length > end) {
            throw new IllegalArgumentException("Truncated packed field in ONNX protobuf");
        }
        int fieldEnd = pos + length;
        java.util.List<Float> out = new java.util.ArrayList<>();
        while (pos < fieldEnd) {
            out.add(readFloat());
        }
        if (pos != fieldEnd) {
            throw new IllegalArgumentException("Packed float field overrun");
        }
        return List.copyOf(out);
    }

    public List<Double> readPackedDouble() {
        int length = (int) readVarint();
        if (pos + length > end) {
            throw new IllegalArgumentException("Truncated packed field in ONNX protobuf");
        }
        int fieldEnd = pos + length;
        java.util.List<Double> out = new java.util.ArrayList<>();
        while (pos < fieldEnd) {
            out.add(readDouble());
        }
        if (pos != fieldEnd) {
            throw new IllegalArgumentException("Packed double field overrun");
        }
        return List.copyOf(out);
    }

    public ProtoReader readMessage() {
        int length = (int) readVarint();
        if (pos + length > end) {
            throw new IllegalArgumentException("Truncated sub-message in ONNX protobuf");
        }
        ProtoReader sub = new ProtoReader(data, pos, length);
        pos += length;
        return sub;
    }

    public void skipField(int wireType) {
        switch (wireType) {
            case 0 -> readVarint();
            case 1 -> {
                skipBytes(8);
            }
            case 5 -> {
                skipBytes(4);
            }
            case 2 -> {
                int length = (int) readVarint();
                skipBytes(length);
            }
            default -> throw new IllegalArgumentException("Unsupported protobuf wire type " + wireType);
        }
    }

    private void skipBytes(int count) {
        if (pos + count > end) {
            throw new IllegalArgumentException("Truncated fixed field in ONNX protobuf");
        }
        pos += count;
    }

    private int readFixed32() {
        if (pos + 4 > end) {
            throw new IllegalArgumentException("Truncated 32-bit field");
        }
        int value = (data[pos] & 0xFF) | ((data[pos + 1] & 0xFF) << 8) | ((data[pos + 2] & 0xFF) << 16)
                | ((data[pos + 3] & 0xFF) << 24);
        pos += 4;
        return value;
    }

    private long readFixed64() {
        if (pos + 8 > end) {
            throw new IllegalArgumentException("Truncated 64-bit field");
        }
        long value = 0;
        for (int i = 0; i < 8; i++) {
            value |= (long) (data[pos + i] & 0xFF) << (8 * i);
        }
        pos += 8;
        return value;
    }
}
