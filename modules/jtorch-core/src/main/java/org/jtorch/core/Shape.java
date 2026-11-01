package org.jtorch.core;

import java.util.Arrays;
import java.util.List;

/** Immutable, defensively copied tensor shape with overflow-checked element counts. */
public final class Shape {
    private static final Shape SCALAR = new Shape(new long[0]);

    private final long[] dims;
    private final long numel;

    private Shape(long[] dims) {
        for (long dim : dims) {
            if (dim < 0) {
                throw new ShapeException("Negative dimension in shape " + Arrays.toString(dims));
            }
        }
        this.dims = dims;
        this.numel = checkedNumel(dims);
    }

    public static Shape of(long... dims) {
        return dims.length == 0 ? SCALAR : new Shape(dims.clone());
    }

    public static Shape of(List<Long> dims) {
        long[] copy = new long[dims.size()];
        for (int i = 0; i < copy.length; i++) {
            Long value = dims.get(i);
            if (value == null) {
                throw new ShapeException("Null dimension in shape");
            }
            copy[i] = value;
        }
        return of(copy);
    }

    public static Shape scalar() {
        return SCALAR;
    }

    /** Builds a shape from interleaved size/stride pairs used by the internal APIs. */
    public static Shape fromArray(long[] dims) {
        return of(dims);
    }

    private static long checkedNumel(long[] dims) {
        long total = 1;
        for (long dim : dims) {
            try {
                total = Math.multiplyExact(total, dim);
            } catch (ArithmeticException error) {
                throw new ShapeException("Shape " + Arrays.toString(dims) + " numel overflows a 64-bit size");
            }
        }
        return total;
    }

    public int rank() {
        return dims.length;
    }

    public long dim(int index) {
        return dims[normalizeDim(index)];
    }

    public boolean isScalar() {
        return dims.length == 0;
    }

    public long numel() {
        return numel;
    }

    public long[] toArray() {
        return dims.clone();
    }

    /** Internal accessor; callers must not mutate the returned array. */
    long[] raw() {
        return dims;
    }

    public int normalizeDim(int dim) {
        int rank = dims.length;
        if (dim < -rank || dim >= rank) {
            throw new IndexOutOfBoundsException(
                    "Dimension out of range (expected to be in range of [" + (-rank) + ", " + (rank - 1)
                            + "], but got " + dim + ")");
        }
        return dim < 0 ? dim + rank : dim;
    }

    /** PyTorch-style right-aligned broadcasting. */
    public static Shape broadcast(Shape a, Shape b) {
        int rank = Math.max(a.rank(), b.rank());
        long[] out = new long[rank];
        for (int i = 0; i < rank; i++) {
            long left = i < rank - a.rank() ? 1 : a.dims[i - (rank - a.rank())];
            long right = i < rank - b.rank() ? 1 : b.dims[i - (rank - b.rank())];
            if (left == right || right == 1) {
                out[i] = left;
            } else if (left == 1) {
                out[i] = right;
            } else {
                throw new ShapeException(
                        "The size of tensor a (" + a + ") must match the size of tensor b (" + b
                                + ") at non-singleton dimension " + i);
            }
        }
        return of(out);
    }

    /** Removes a dimension, requiring it to have size {@code size}. */
    public Shape withoutDim(int dim) {
        int normalized = normalizeDim(dim);
        long[] out = new long[dims.length - 1];
        System.arraycopy(dims, 0, out, 0, normalized);
        System.arraycopy(dims, normalized + 1, out, normalized, dims.length - normalized - 1);
        return of(out);
    }

    public Shape withDim(int dim, long size) {
        if (dim < 0 || dim > dims.length) {
            throw new IndexOutOfBoundsException("Insert dimension out of range: " + dim);
        }
        long[] out = new long[dims.length + 1];
        System.arraycopy(dims, 0, out, 0, dim);
        out[dim] = size;
        System.arraycopy(dims, dim, out, dim + 1, dims.length - dim);
        return of(out);
    }

    public Shape withSize(int dim, long size) {
        int normalized = normalizeDim(dim);
        long[] out = dims.clone();
        out[normalized] = size;
        return of(out);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Shape shape && Arrays.equals(dims, shape.dims);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }

    @Override
    public String toString() {
        StringBuilder text = new StringBuilder("torch.Size([");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) {
                text.append(", ");
            }
            text.append(dims[i]);
        }
        return text.append("])").toString();
    }
}
