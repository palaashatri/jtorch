package org.jtorch.autograd;

import java.util.Arrays;
import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.OpRegistry;
import org.jtorch.core.Scalar;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.value.Value;

/** Gradient arithmetic built only from dispatched operators, so every reduction is itself differentiable. */
final class GradUtil {
    private GradUtil() {}

    static Tensor constant(Tensor like, double value) {
        return Dispatcher.get()
                .call(
                        OpRegistry.FULL,
                        Value.integers(),
                        Value.scalar(Scalar.of(value)),
                        Value.dtype(like.dtype()),
                        Value.device(like.device()))[0]
                .asTensor();
    }

    static Tensor scale(Tensor tensor, double factor) {
        if (factor == 1.0) {
            return tensor;
        }
        return tensor.mul(constant(tensor, factor));
    }

    static Tensor arange(long start, long end, DType dtype, Device device) {
        return Dispatcher.get()
                .call(
                        OpRegistry.ARANGE,
                        Value.scalar(Scalar.of(start)),
                        Value.scalar(Scalar.of(end)),
                        Value.scalar(Scalar.of(1)),
                        Value.dtype(dtype),
                        Value.device(device))[0]
                .asTensor();
    }

    static int[] normalizeDims(Tensor tensor, long[] dims) {
        int[] normalized = new int[dims.length];
        boolean[] seen = new boolean[tensor.dim()];
        for (int i = 0; i < dims.length; i++) {
            int dim = tensor.shape().normalizeDim((int) dims[i]);
            if (seen[dim]) {
                throw new IllegalArgumentException("dim " + dim + " appears multiple times in the reduction list");
            }
            seen[dim] = true;
            normalized[i] = dim;
        }
        Arrays.sort(normalized);
        return normalized;
    }

    static Tensor summation(Tensor gradient, Tensor input, int[] dims, boolean keepdim) {
        Tensor expanded = gradient;
        if (!keepdim) {
            for (int dim : dims) {
                expanded = expanded.unsqueeze(dim);
            }
        }
        return expanded.expand(input.shape().toArray());
    }

    /**
     * Reduces a broadcast gradient back to {@code targetShape}, summing over leading and size-1 dimensions in the
     * same order PyTorch's broadcast backward does.
     */
    static Tensor reduceToShape(Tensor gradient, Shape targetShape) {
        if (gradient.shape().equals(targetShape)) {
            return gradient;
        }
        if (gradient.dim() < targetShape.rank()) {
            throw new IllegalStateException("Gradient rank " + gradient.dim() + " is lower than target rank "
                    + targetShape.rank());
        }
        Tensor result = gradient;
        int extra = gradient.dim() - targetShape.rank();
        if (extra > 0) {
            int[] leading = new int[extra];
            for (int i = 0; i < extra; i++) {
                leading[i] = i;
            }
            result = result.sum(leading, false);
        }
        int[] singletonDims = new int[targetShape.rank()];
        int count = 0;
        for (int i = 0; i < targetShape.rank(); i++) {
            if (targetShape.dim(i) == 1 && result.size(i) != 1) {
                singletonDims[count++] = i;
            }
        }
        if (count > 0) {
            result = result.sum(Arrays.copyOf(singletonDims, count), true);
        }
        if (!result.shape().equals(targetShape)) {
            result = result.reshape(targetShape.toArray());
        }
        return result;
    }

    static Tensor oneHot(Tensor target, int classes, DType dtype) {
        Tensor range = arange(0, classes, DType.INT64, target.device());
        Tensor reshaped = range.reshape(1, classes);
        Tensor labels = target.dim() == 1 ? target.unsqueeze(1) : target;
        return reshaped.eq(labels).to(dtype);
    }

    /** Gradients of {@code a @ b} with NumPy batch broadcasting, including 1-D vector cases. */
    static Tensor[] matmulGrads(Tensor a, Tensor b, Tensor gradient) {
        int aRank = a.dim();
        int bRank = b.dim();
        Tensor a2 = aRank == 1 ? a.reshape(1, a.size(0)) : a;
        Tensor b2 = bRank == 1 ? b.reshape(b.size(0), 1) : b;
        Tensor g2 = gradient;
        if (aRank == 1 && bRank == 1) {
            g2 = gradient.reshape(1, 1);
        } else if (aRank == 1) {
            g2 = gradient.unsqueeze(-2);
        } else if (bRank == 1) {
            g2 = gradient.unsqueeze(-1);
        }
        Tensor gradA = g2.matmul(b2.transpose(-1, -2));
        gradA = reduceToShape(gradA, a2.shape()).reshape(a.shape().toArray());
        Tensor gradB = a2.transpose(-1, -2).matmul(g2);
        gradB = reduceToShape(gradB, b2.shape()).reshape(b.shape().toArray());
        return new Tensor[] {gradA, gradB};
    }
}
