package org.jtorch.serialization;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.jtorch.core.memory.TensorAccess;

/** Little-endian, C-order tensor byte codec shared by the checkpoint and Safetensors formats. */
final class TensorBytes {
    private TensorBytes() {}

    static byte[] encode(Tensor tensor) {
        long numel = tensor.numel();
        if (numel > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor has too many elements for this format: " + numel);
        }
        int elementSize = tensor.dtype().sizeBytes();
        ByteBuffer buffer = ByteBuffer.allocate((int) numel * elementSize).order(ByteOrder.LITTLE_ENDIAN);
        for (long i = 0; i < numel; i++) {
            long offset = tensor.elementOffset(i);
            switch (tensor.dtype()) {
                case FLOAT64 -> buffer.putDouble(TensorAccess.getDouble(tensor.impl(), offset));
                case FLOAT32 -> buffer.putFloat(TensorAccess.getFloat(tensor.impl(), offset));
                case INT64 -> buffer.putLong(TensorAccess.getLong(tensor.impl(), offset));
                case INT32 -> buffer.putInt((int) TensorAccess.getLong(tensor.impl(), offset));
                case BOOL -> buffer.put(TensorAccess.getBoolean(tensor.impl(), offset) ? (byte) 1 : (byte) 0);
            }
        }
        return buffer.array();
    }

    static Tensor decode(byte[] data, int offset, int length, DType dtype, Shape shape) {
        long expected = shape.numel() * dtype.sizeBytes();
        if (length != expected) {
            throw new SerializationException("Tensor payload has " + length + " bytes but " + shape + " " + dtype
                    + " needs " + expected);
        }
        Tensor tensor = Torch.empty(shape, TensorOptions.defaults().dtype(dtype));
        ByteBuffer buffer = ByteBuffer.wrap(data, offset, length).order(ByteOrder.LITTLE_ENDIAN);
        for (long i = 0; i < shape.numel(); i++) {
            long element = tensor.elementOffset(i);
            switch (dtype) {
                case FLOAT64 -> TensorAccess.setDouble(tensor.impl(), element, buffer.getDouble());
                case FLOAT32 -> TensorAccess.setFloat(tensor.impl(), element, buffer.getFloat());
                case INT64 -> TensorAccess.setLong(tensor.impl(), element, buffer.getLong());
                case INT32 -> TensorAccess.setLong(tensor.impl(), element, buffer.getInt());
                case BOOL -> TensorAccess.setBoolean(tensor.impl(), element, buffer.get() != 0);
            }
        }
        return tensor;
    }

    static String safetensorsName(DType dtype) {
        return switch (dtype) {
            case FLOAT64 -> "F64";
            case FLOAT32 -> "F32";
            case INT64 -> "I64";
            case INT32 -> "I32";
            case BOOL -> "BOOL";
        };
    }

    static DType fromSafetensorsName(String name) {
        return switch (name) {
            case "F64" -> DType.FLOAT64;
            case "F32" -> DType.FLOAT32;
            case "I64" -> DType.INT64;
            case "I32" -> DType.INT32;
            case "BOOL" -> DType.BOOL;
            default -> throw new SerializationException("Unsupported Safetensors dtype '" + name + "'");
        };
    }

}
