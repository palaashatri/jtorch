package org.jtorch.onnx;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.jtorch.core.memory.TensorAccess;

/** Decodes ONNX TensorProto (raw_data or typed repeated data) into JTorch tensors. */
final class OnnxTensorDecoder {
    private OnnxTensorDecoder() {}

    static Tensor decode(OnnxProto.Tensor proto) {
        if (proto.isExternal()) {
            throw new OnnxException("ONNX external tensor data is not supported in this build: " + proto.name());
        }
        DType dtype = mapDataType(proto.dataType());
        long[] dims = new long[proto.dims().size()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = proto.dims().get(i);
        }
        Shape shape = dims.length == 0 ? Shape.scalar() : Shape.of(dims);
        long expected = shape.numel();
        Tensor tensor = Torch.empty(shape, TensorOptions.defaults().dtype(dtype));
        if (proto.rawData() != null) {
            decodeRaw(tensor, dtype, proto.rawData(), expected);
            return tensor;
        }
        if (dtype == DType.FLOAT64) {
            if (proto.doubleData().size() != expected) {
                throw new OnnxException("ONNX tensor " + proto.name() + " double data length mismatch");
            }
            for (int i = 0; i < expected; i++) {
                TensorAccess.setDouble(tensor.impl(), tensor.elementOffset(i), proto.doubleData().get(i));
            }
        } else if (dtype == DType.FLOAT32) {
            if (proto.floatData().size() != expected) {
                throw new OnnxException("ONNX tensor " + proto.name() + " float data length mismatch");
            }
            for (int i = 0; i < expected; i++) {
                TensorAccess.setFloat(tensor.impl(), tensor.elementOffset(i), proto.floatData().get(i));
            }
        } else if (dtype == DType.INT64) {
            if (proto.int64Data().size() != expected) {
                throw new OnnxException("ONNX tensor " + proto.name() + " int64 data length mismatch");
            }
            for (int i = 0; i < expected; i++) {
                TensorAccess.setLong(tensor.impl(), tensor.elementOffset(i), proto.int64Data().get(i));
            }
        } else if (dtype == DType.INT32) {
            if (proto.int32Data().size() != expected) {
                throw new OnnxException("ONNX tensor " + proto.name() + " int32 data length mismatch");
            }
            for (int i = 0; i < expected; i++) {
                TensorAccess.setLong(tensor.impl(), tensor.elementOffset(i), proto.int32Data().get(i));
            }
        } else {
            throw new OnnxException("ONNX tensor " + proto.name() + " type " + proto.dataType()
                    + " has no typed data in this build");
        }
        return tensor;
    }

    static DType mapDataType(long code) {
        return switch ((int) code) {
            case 1 -> DType.FLOAT32;
            case 11 -> DType.FLOAT64;
            case 6 -> DType.INT32;
            case 7 -> DType.INT64;
            case 9 -> DType.BOOL;
            case 14 -> DType.COMPLEX64;
            case 15 -> DType.COMPLEX128;
            default -> throw new OnnxException("ONNX tensor data_type " + code + " is not supported in this build");
        };
    }

    private static void decodeRaw(Tensor tensor, DType dtype, byte[] raw, long expected) {
        ByteBuffer buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        long needed = expected * dtype.sizeBytes();
        if (raw.length != needed) {
            throw new OnnxException("ONNX raw tensor has " + raw.length + " bytes but " + tensor.shape() + " "
                    + dtype + " needs " + needed);
        }
        for (long i = 0; i < expected; i++) {
            long offset = tensor.elementOffset(i);
            switch (dtype) {
                case FLOAT64 -> TensorAccess.setDouble(tensor.impl(), offset, buffer.getDouble());
                case FLOAT32 -> TensorAccess.setFloat(tensor.impl(), offset, buffer.getFloat());
                case INT64 -> TensorAccess.setLong(tensor.impl(), offset, buffer.getLong());
                case INT32 -> TensorAccess.setLong(tensor.impl(), offset, buffer.getInt());
                case BOOL -> TensorAccess.setBoolean(tensor.impl(), offset, buffer.get() != 0);
                case COMPLEX128 -> TensorAccess.setComplex(tensor.impl(), offset, buffer.getDouble(),
                        buffer.getDouble());
                case COMPLEX64 -> TensorAccess.setComplex(tensor.impl(), offset, buffer.getFloat(),
                        buffer.getFloat());
            }
        }
    }
}
