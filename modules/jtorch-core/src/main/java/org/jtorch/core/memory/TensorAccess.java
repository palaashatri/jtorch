package org.jtorch.core.memory;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.jtorch.core.DType;
import org.jtorch.core.JTorchException;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorImpl;

/** Typed element access for host storage. Meta tensors reject all access. */
public final class TensorAccess {
    private TensorAccess() {}

    public static double getDouble(TensorImpl tensor, long elementOffset) {
        MemorySegment segment = tensor.segment();
        DType dtype = tensor.dtype();
        long byteOffset = elementOffset * dtype.sizeBytes();
        return switch (dtype) {
            case FLOAT64 -> segment.get(ValueLayout.JAVA_DOUBLE, byteOffset);
            case FLOAT32 -> segment.get(ValueLayout.JAVA_FLOAT, byteOffset);
            case INT64 -> segment.get(ValueLayout.JAVA_LONG, byteOffset);
            case INT32 -> segment.get(ValueLayout.JAVA_INT, byteOffset);
            case BOOL -> segment.get(ValueLayout.JAVA_BYTE, byteOffset) != 0 ? 1 : 0;
        };
    }

    public static float getFloat(TensorImpl tensor, long elementOffset) {
        if (tensor.dtype() == DType.FLOAT32) {
            return tensor.segment().get(ValueLayout.JAVA_FLOAT, elementOffset * 4);
        }
        return (float) getDouble(tensor, elementOffset);
    }

    public static long getLong(TensorImpl tensor, long elementOffset) {
        MemorySegment segment = tensor.segment();
        DType dtype = tensor.dtype();
        long byteOffset = elementOffset * dtype.sizeBytes();
        return switch (dtype) {
            case INT64 -> segment.get(ValueLayout.JAVA_LONG, byteOffset);
            case INT32 -> segment.get(ValueLayout.JAVA_INT, byteOffset);
            case FLOAT64 -> (long) segment.get(ValueLayout.JAVA_DOUBLE, byteOffset);
            case FLOAT32 -> (long) segment.get(ValueLayout.JAVA_FLOAT, byteOffset);
            case BOOL -> segment.get(ValueLayout.JAVA_BYTE, byteOffset) != 0 ? 1 : 0;
        };
    }

    public static boolean getBoolean(TensorImpl tensor, long elementOffset) {
        DType dtype = tensor.dtype();
        if (dtype == DType.BOOL) {
            return tensor.segment().get(ValueLayout.JAVA_BYTE, elementOffset) != 0;
        }
        return getDouble(tensor, elementOffset) != 0;
    }

    public static void setDouble(TensorImpl tensor, long elementOffset, double value) {
        MemorySegment segment = tensor.segment();
        DType dtype = tensor.dtype();
        long byteOffset = elementOffset * dtype.sizeBytes();
        switch (dtype) {
            case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE, byteOffset, value);
            case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT, byteOffset, (float) value);
            case INT64 -> segment.set(ValueLayout.JAVA_LONG, byteOffset, (long) value);
            case INT32 -> segment.set(ValueLayout.JAVA_INT, byteOffset, (int) (long) value);
            case BOOL -> segment.set(ValueLayout.JAVA_BYTE, byteOffset, value != 0 ? (byte) 1 : (byte) 0);
        }
    }

    public static void setFloat(TensorImpl tensor, long elementOffset, float value) {
        if (tensor.dtype() == DType.FLOAT32) {
            tensor.segment().set(ValueLayout.JAVA_FLOAT, elementOffset * 4, value);
        } else {
            setDouble(tensor, elementOffset, value);
        }
    }

    public static void setLong(TensorImpl tensor, long elementOffset, long value) {
        MemorySegment segment = tensor.segment();
        DType dtype = tensor.dtype();
        long byteOffset = elementOffset * dtype.sizeBytes();
        switch (dtype) {
            case INT64 -> segment.set(ValueLayout.JAVA_LONG, byteOffset, value);
            case INT32 -> segment.set(ValueLayout.JAVA_INT, byteOffset, (int) value);
            case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE, byteOffset, (double) value);
            case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT, byteOffset, (float) value);
            case BOOL -> segment.set(ValueLayout.JAVA_BYTE, byteOffset, value != 0 ? (byte) 1 : (byte) 0);
        }
    }

    public static void setBoolean(TensorImpl tensor, long elementOffset, boolean value) {
        if (tensor.dtype() == DType.BOOL) {
            tensor.segment().set(ValueLayout.JAVA_BYTE, elementOffset, value ? (byte) 1 : (byte) 0);
        } else {
            setDouble(tensor, elementOffset, value ? 1 : 0);
        }
    }

    /** Computes the storage element offset for a multi-index. */
    public static long offsetOf(TensorImpl tensor, long[] indices) {
        if (indices.length != tensor.sizes().rank()) {
            throw new JTorchException("Index rank mismatch");
        }
        long offset = tensor.storageOffset();
        for (int i = 0; i < indices.length; i++) {
            offset += indices[i] * tensor.strides().stride(i);
        }
        return offset;
    }
}
