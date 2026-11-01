package org.jtorch.cpu;

import java.util.Arrays;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorImpl;

/** Shape, stride and dtype rules shared by meta and CPU kernels. */
final class ShapeRules {
    private ShapeRules() {}

    static Shape broadcastShape(Tensor left, Tensor right) {
        return Shape.broadcast(left.shape(), right.shape());
    }

    static Shape broadcastShape(Shape left, Shape right) {
        return Shape.broadcast(left, right);
    }

    static DType binaryDtype(Tensor left, Tensor right) {
        return DType.resultType(left.dtype(), right.dtype());
    }

    static DType scalarDtype(Tensor tensor, org.jtorch.core.Scalar scalar) {
        return DType.resultTypeWithScalar(tensor.dtype(), scalar);
    }

    /** True-division dtype rule: integer operands produce the default float dtype. */
    static DType divisionDtype(Tensor left, Tensor right) {
        DType promoted = binaryDtype(left, right);
        return promoted.isFloatingPoint() ? promoted : DType.defaultFloat();
    }

    /** Dtypes that require promotion to the default float for transcendental math. */
    static DType floatMathDtype(DType dtype) {
        return dtype.isFloatingPoint() ? dtype : DType.defaultFloat();
    }

    /** Integral reductions promote to int64, matching PyTorch's sum/prod accumulation. */
    static DType integralPromotion(DType dtype) {
        return dtype.isFloatingPoint() ? dtype : DType.INT64;
    }

    static int[] normalizeDims(Shape shape, long[] dims) {
        int[] normalized = new int[dims.length];
        boolean[] seen = new boolean[shape.rank()];
        for (int i = 0; i < dims.length; i++) {
            long raw = dims[i];
            if (raw < -shape.rank() || raw >= shape.rank()) {
                throw new IndexOutOfBoundsException("Dimension out of range: " + raw + " for rank " + shape.rank());
            }
            int dim = raw < 0 ? (int) raw + shape.rank() : (int) raw;
            if (seen[dim]) {
                throw new IllegalArgumentException("dim " + dim + " appears multiple times in the reduction list");
            }
            seen[dim] = true;
            normalized[i] = dim;
        }
        Arrays.sort(normalized);
        return normalized;
    }

    static Shape reductionShape(Shape input, int[] dims, boolean keepdim) {
        long[] out = input.toArray();
        if (keepdim) {
            for (int dim : dims) {
                out[dim] = 1;
            }
        } else {
            long[] reduced = new long[input.rank() - dims.length];
            int index = 0;
            outer:
            for (int i = 0; i < input.rank(); i++) {
                for (int dim : dims) {
                    if (dim == i) {
                        continue outer;
                    }
                }
                reduced[index++] = out[i];
            }
            out = reduced;
        }
        return Shape.of(out);
    }

    static Shape keepDimShape(Shape input, int[] dims) {
        long[] out = input.toArray();
        for (int dim : dims) {
            out[dim] = 1;
        }
        return Shape.of(out);
    }

    static long reductionCount(Shape input, int[] dims) {
        long count = 1;
        for (int dim : dims) {
            count *= input.dim(dim);
        }
        return count;
    }

    /** Effective right-aligned broadcast strides for one input against an output shape. */
    static long[] broadcastStrides(TensorImpl input, Shape output) {
        int outRank = output.rank();
        int rank = input.sizes().rank();
        long[] strides = new long[outRank];
        for (int i = 0; i < outRank; i++) {
            int inputDim = i - (outRank - rank);
            if (inputDim < 0) {
                strides[i] = 0;
            } else if (input.sizes().dim(inputDim) == 1 && output.dim(i) != 1) {
                strides[i] = 0;
            } else {
                if (input.sizes().dim(inputDim) != output.dim(i)) {
                    throw new ShapeException("The size of tensor input (" + input.sizes().dim(inputDim)
                            + ") must match the output size (" + output.dim(i) + ") at dimension " + i);
                }
                strides[i] = input.strides().stride(inputDim);
            }
        }
        return strides;
    }

    /** Shape of {@code matmul} following PyTorch's vector special cases. */
    static Shape matmulShape(Shape left, Shape right) {
        if (left.rank() == 0 || right.rank() == 0) {
            throw new ShapeException("matmul does not accept 0-dimensional operands");
        }
        long k = left.dim(left.rank() - 1);
        long otherK = right.rank() == 1 ? right.dim(0) : right.dim(right.rank() - 2);
        if (k != otherK) {
            throw new ShapeException("mat1 and mat2 shapes cannot be multiplied (" + left + " and " + right + ")");
        }
        long m = left.rank() == 1 ? 1 : left.dim(left.rank() - 2);
        long n = right.rank() == 1 ? 1 : right.dim(right.rank() - 1);
        Shape leftBatch = left.rank() <= 2 ? Shape.scalar() : prefix(left, left.rank() - 2);
        Shape rightBatch = right.rank() <= 2 ? Shape.scalar() : prefix(right, right.rank() - 2);
        Shape batch = Shape.broadcast(leftBatch, rightBatch);
        long[] dims;
        int offset = 0;
        if (left.rank() == 1 && right.rank() == 1) {
            dims = new long[0];
        } else if (left.rank() == 1) {
            dims = new long[batch.rank() + 1];
            System.arraycopy(batch.toArray(), 0, dims, 0, batch.rank());
            dims[batch.rank()] = n;
        } else if (right.rank() == 1) {
            dims = new long[batch.rank() + 1];
            System.arraycopy(batch.toArray(), 0, dims, 0, batch.rank());
            dims[batch.rank()] = m;
        } else {
            dims = new long[batch.rank() + 2];
            System.arraycopy(batch.toArray(), 0, dims, 0, batch.rank());
            dims[batch.rank()] = m;
            dims[batch.rank() + 1] = n;
        }
        return Shape.of(dims);
    }

    private static Shape prefix(Shape shape, int rank) {
        long[] dims = new long[rank];
        System.arraycopy(shape.toArray(), 0, dims, 0, rank);
        return Shape.of(dims);
    }

    static void requireFloating(Tensor tensor, String op) {
        if (!tensor.dtype().isFloatingPoint()) {
            throw new org.jtorch.core.DTypeException(op + " expects a floating point tensor, got " + tensor.dtype());
        }
    }

    static void requireScaling(Tensor self, String op) {
        if (self.requiresGrad() && org.jtorch.core.GradMode.isEnabled()) {
            throw new org.jtorch.core.AutogradException(
                    "a tensor that requires grad is being used in an in-place operation (" + op
                            + "); wrap the call in GradMode.noGrad or mutate a detached tensor");
        }
    }
}
