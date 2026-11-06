package org.jtorch.core.memory;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.jtorch.core.DType;
import org.jtorch.core.DTypeException;
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
            case COMPLEX64, COMPLEX128 ->
                throw new DTypeException("Complex tensors have no single double value; use getComplexReal/Imaginary");
        };
    }

    public static double getComplexReal(TensorImpl tensor, long elementOffset) {
        MemorySegment segment = tensor.segment();
        long byteOffset = elementOffset * tensor.dtype().sizeBytes();
        return switch (tensor.dtype()) {
            case COMPLEX128 -> segment.get(ValueLayout.JAVA_DOUBLE, byteOffset);
            case COMPLEX64 -> segment.get(ValueLayout.JAVA_FLOAT, byteOffset);
            default -> throw new DTypeException("getComplexReal requires a complex dtype, got " + tensor.dtype());
        };
    }

    public static double getComplexImaginary(TensorImpl tensor, long elementOffset) {
        MemorySegment segment = tensor.segment();
        long byteOffset = elementOffset * tensor.dtype().sizeBytes();
        return switch (tensor.dtype()) {
            case COMPLEX128 -> segment.get(ValueLayout.JAVA_DOUBLE, byteOffset + 8);
            case COMPLEX64 -> segment.get(ValueLayout.JAVA_FLOAT, byteOffset + 4);
            default -> throw new DTypeException("getComplexImaginary requires a complex dtype, got " + tensor.dtype());
        };
    }

    public static void setComplex(TensorImpl tensor, long elementOffset, double real, double imaginary) {
        MemorySegment segment = tensor.segment();
        long byteOffset = elementOffset * tensor.dtype().sizeBytes();
        switch (tensor.dtype()) {
            case COMPLEX128 -> {
                segment.set(ValueLayout.JAVA_DOUBLE, byteOffset, real);
                segment.set(ValueLayout.JAVA_DOUBLE, byteOffset + 8, imaginary);
            }
            case COMPLEX64 -> {
                segment.set(ValueLayout.JAVA_FLOAT, byteOffset, (float) real);
                segment.set(ValueLayout.JAVA_FLOAT, byteOffset + 4, (float) imaginary);
            }
            default -> throw new DTypeException("setComplex requires a complex dtype, got " + tensor.dtype());
        }
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
            case COMPLEX64, COMPLEX128 -> throw new DTypeException("Complex tensors have no single long value");
        };
    }

    public static boolean getBoolean(TensorImpl tensor, long elementOffset) {
        DType dtype = tensor.dtype();
        if (dtype == DType.BOOL) {
            return tensor.segment().get(ValueLayout.JAVA_BYTE, elementOffset) != 0;
        }
        if (dtype.isComplex()) {
            throw new DTypeException("Complex tensors have no single boolean value");
        }
        return getDouble(tensor, elementOffset) != 0;
    }

    /** Real scalar promotion: complex targets receive (value, 0), matching wrapped scalar semantics. */
    public static void setDouble(TensorImpl tensor, long elementOffset, double value) {
        if (tensor.dtype().isComplex()) {
            setComplex(tensor, elementOffset, value, 0.0);
            return;
        }
        MemorySegment segment = tensor.segment();
        DType dtype = tensor.dtype();
        long byteOffset = elementOffset * dtype.sizeBytes();
        switch (dtype) {
            case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE, byteOffset, value);
            case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT, byteOffset, (float) value);
            case INT64 -> segment.set(ValueLayout.JAVA_LONG, byteOffset, (long) value);
            case INT32 -> segment.set(ValueLayout.JAVA_INT, byteOffset, (int) (long) value);
            case BOOL -> segment.set(ValueLayout.JAVA_BYTE, byteOffset, value != 0 ? (byte) 1 : (byte) 0);
            case COMPLEX64, COMPLEX128 -> throw new IllegalStateException("complex handled above");
        }
    }

    public static void setFloat(TensorImpl tensor, long elementOffset, float value) {
        if (tensor.dtype() == DType.FLOAT32) {
            tensor.segment().set(ValueLayout.JAVA_FLOAT, elementOffset * 4, value);
        } else if (tensor.dtype().isComplex()) {
            setComplex(tensor, elementOffset, value, 0.0);
        } else {
            setDouble(tensor, elementOffset, value);
        }
    }

    public static void setLong(TensorImpl tensor, long elementOffset, long value) {
        if (tensor.dtype().isComplex()) {
            setComplex(tensor, elementOffset, (double) value, 0.0);
            return;
        }
        MemorySegment segment = tensor.segment();
        DType dtype = tensor.dtype();
        long byteOffset = elementOffset * dtype.sizeBytes();
        switch (dtype) {
            case INT64 -> segment.set(ValueLayout.JAVA_LONG, byteOffset, value);
            case INT32 -> segment.set(ValueLayout.JAVA_INT, byteOffset, (int) value);
            case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE, byteOffset, (double) value);
            case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT, byteOffset, (float) value);
            case BOOL -> segment.set(ValueLayout.JAVA_BYTE, byteOffset, value != 0 ? (byte) 1 : (byte) 0);
            case COMPLEX64, COMPLEX128 -> throw new IllegalStateException("complex handled above");
        }
    }

    public static void setBoolean(TensorImpl tensor, long elementOffset, boolean value) {
        if (tensor.dtype() == DType.BOOL) {
            tensor.segment().set(ValueLayout.JAVA_BYTE, elementOffset, value ? (byte) 1 : (byte) 0);
        } else if (tensor.dtype().isComplex()) {
            setComplex(tensor, elementOffset, value ? 1 : 0, 0.0);
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
