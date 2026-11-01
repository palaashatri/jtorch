package org.jtorch.core;

import java.util.Arrays;

/** Immutable, defensively copied tensor strides measured in elements (not bytes). */
public final class Strides {
    private static final Strides EMPTY = new Strides(new long[0]);

    private final long[] strides;

    private Strides(long[] strides) {
        for (long stride : strides) {
            if (stride < 0) {
                throw new ShapeException("Negative stride in " + Arrays.toString(strides));
            }
        }
        this.strides = strides;
    }

    public static Strides of(long... strides) {
        return strides.length == 0 ? EMPTY : new Strides(strides.clone());
    }

    public static Strides empty() {
        return EMPTY;
    }

    /** Row-major contiguous strides for the given shape. */
    public static Strides contiguous(Shape shape) {
        long[] dims = shape.raw();
        long[] result = new long[dims.length];
        long running = 1;
        for (int i = dims.length - 1; i >= 0; i--) {
            result[i] = running;
            running *= Math.max(dims[i], 1);
        }
        return new Strides(result);
    }

    /** PyTorch contiguity: size-1 dimensions may carry any stride, and empty tensors are contiguous. */
    public boolean isContiguous(Shape shape) {
        if (shape.numel() == 0) {
            return true;
        }
        long expected = 1;
        long[] dims = shape.raw();
        for (int i = dims.length - 1; i >= 0; i--) {
            if (dims[i] == 0) {
                return true;
            }
            if (dims[i] != 1) {
                if (strides[i] != expected) {
                    return false;
                }
                expected *= dims[i];
            }
        }
        return true;
    }

    public int rank() {
        return strides.length;
    }

    public long stride(int dim) {
        return strides[dim];
    }

    public long[] toArray() {
        return strides.clone();
    }

    long[] raw() {
        return strides;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Strides value && Arrays.equals(strides, value.strides);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(strides);
    }

    @Override
    public String toString() {
        StringBuilder text = new StringBuilder("(");
        for (int i = 0; i < strides.length; i++) {
            if (i > 0) {
                text.append(", ");
            }
            text.append(strides[i]);
        }
        return text.append(")").toString();
    }
}
