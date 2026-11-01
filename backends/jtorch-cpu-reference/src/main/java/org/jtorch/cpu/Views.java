package org.jtorch.cpu;

import java.util.Arrays;
import org.jtorch.core.Device;
import org.jtorch.core.Scalar;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Strides;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorImpl;

/** View geometry: one implementation of reshape/transpose/expand rules used by both meta and CPU kernels. */
final class Views {
    private Views() {}

    record ViewSpec(Shape shape, Strides strides, long offset, boolean requiresCopy) {}

    static ViewSpec reshape(Tensor input, long[] requested, boolean strict) {
        Shape shape = Shape.of(resolveShape(requested, input.numel()));
        if (shape.numel() != input.numel()) {
            throw new ShapeException("shape " + shape + " is invalid for input of size " + input.numel());
        }
        if (shape.equals(input.shape())) {
            return new ViewSpec(shape, input.strides(), input.storageOffset(), false);
        }
        // A view may drop size-1 dimensions: transposed.unsqueeze(0).view(3, 2) is legal even though the
        // 3-D tensor is not contiguous.
        long[] squeezedShape = new long[input.dim()];
        long[] squeezedStrides = new long[input.dim()];
        int squeezedRank = 0;
        for (int i = 0; i < input.dim(); i++) {
            if (input.size(i) != 1) {
                squeezedShape[squeezedRank] = input.size(i);
                squeezedStrides[squeezedRank] = input.stride(i);
                squeezedRank++;
            }
        }
        if (shape.rank() == squeezedRank
                && shape.equals(Shape.of(java.util.Arrays.copyOf(squeezedShape, squeezedRank)))) {
            return new ViewSpec(shape, Strides.of(java.util.Arrays.copyOf(squeezedStrides, squeezedRank)),
                    input.storageOffset(), false);
        }
        if (input.isContiguous() || input.numel() <= 1) {
            return new ViewSpec(shape, Strides.contiguous(shape), input.storageOffset(), false);
        }
        long[] computed = computeStride(input.shape(), input.strides(), shape);
        if (computed != null) {
            return new ViewSpec(shape, Strides.of(computed), input.storageOffset(), false);
        }
        if (strict) {
            throw new ShapeException("view size is not compatible with input tensor's size and stride; call reshape "
                    + "to copy instead");
        }
        return new ViewSpec(shape, Strides.contiguous(shape), 0, true);
    }

    /**
     * PyTorch's {@code computeStride} viewability test: coalesces contiguous chunks of the old geometry and checks
     * that they partition exactly into the requested shape. Returns {@code null} when a copy is required.
     */
    private static long[] computeStride(Shape oldShape, Strides oldStrides, Shape newShape) {
        if (oldShape.numel() == 0 || newShape.numel() == 0) {
            return Strides.contiguous(newShape).toArray();
        }
        if (oldShape.rank() == 0) {
            return Strides.contiguous(newShape).toArray();
        }
        for (int i = 0; i < oldShape.rank(); i++) {
            if (oldShape.dim(i) > 1 && oldStrides.stride(i) == 0) {
                return null;
            }
        }
        long[] newStrides = new long[newShape.rank()];
        int viewDim = newShape.rank() - 1;
        long chunkBaseStride = oldStrides.stride(oldShape.rank() - 1);
        long tensorNumel = 1;
        long viewNumel = 1;
        for (int tensorDim = oldShape.rank() - 1; tensorDim >= 0; tensorDim--) {
            tensorNumel *= oldShape.dim(tensorDim);
            boolean endChunk = tensorDim == 0
                    || (oldShape.dim(tensorDim - 1) != 1
                            && oldStrides.stride(tensorDim - 1) != tensorNumel * chunkBaseStride);
            if (!endChunk) {
                continue;
            }
            while (viewDim >= 0 && (viewNumel < tensorNumel || newShape.dim(viewDim) == 1)) {
                newStrides[viewDim] = viewNumel * chunkBaseStride;
                viewNumel *= newShape.dim(viewDim);
                viewDim--;
            }
            if (viewNumel != tensorNumel) {
                return null;
            }
            if (tensorDim > 0) {
                chunkBaseStride = oldStrides.stride(tensorDim - 1);
                tensorNumel = 1;
                viewNumel = 1;
            }
        }
        return viewDim == -1 ? newStrides : null;
    }

    /** Resolves PyTorch's single inferred {@code -1} dimension. */
    static long[] resolveShape(long[] requested, long numel) {
        long[] target = requested.clone();
        int inferred = -1;
        long known = 1;
        for (int i = 0; i < target.length; i++) {
            if (target[i] == -1) {
                if (inferred >= 0) {
                    throw new ShapeException("only one dimension can be inferred");
                }
                inferred = i;
            } else if (target[i] < 0) {
                throw new ShapeException("invalid shape dimension " + target[i]);
            } else {
                known *= target[i];
            }
        }
        if (inferred >= 0) {
            if (known == 0 || numel % known != 0) {
                throw new ShapeException("shape " + java.util.Arrays.toString(requested)
                        + " is invalid for input of size " + numel);
            }
            target[inferred] = numel / known;
        }
        return target;
    }

    /** Contiguous-stride spec for a copy of the input, used by reshape on non-contiguous tensors. */
    static ViewSpec copied(Shape shape) {
        return new ViewSpec(shape, Strides.contiguous(shape), 0, false);
    }

    static ViewSpec transpose(Tensor input, int dim0, int dim1) {
        if (input.dim() == 0) {
            if (dim0 != 0 || dim1 != 0) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            return new ViewSpec(input.shape(), input.strides(), input.storageOffset(), false);
        }
        int a = input.shape().normalizeDim(dim0);
        int b = input.shape().normalizeDim(dim1);
        long[] sizes = input.shape().toArray();
        long[] strides = input.strides().toArray();
        long swapSize = sizes[a];
        sizes[a] = sizes[b];
        sizes[b] = swapSize;
        long swapStride = strides[a];
        strides[a] = strides[b];
        strides[b] = swapStride;
        return new ViewSpec(Shape.of(sizes), Strides.of(strides), input.storageOffset(), false);
    }

    static ViewSpec permute(Tensor input, long[] dims) {
        boolean[] seen = new boolean[input.dim()];
        long[] sizes = new long[input.dim()];
        long[] strides = new long[input.dim()];
        for (int i = 0; i < dims.length; i++) {
            int dim = input.shape().normalizeDim((int) dims[i]);
            if (seen[dim]) {
                throw new ShapeException("Permutation contains duplicate dimension " + dim);
            }
            seen[dim] = true;
            sizes[i] = input.size(dim);
            strides[i] = input.stride(dim);
        }
        if (dims.length != input.dim()) {
            throw new ShapeException("Permutation length " + dims.length + " does not match rank " + input.dim());
        }
        return new ViewSpec(Shape.of(sizes), Strides.of(strides), input.storageOffset(), false);
    }

    static ViewSpec narrow(Tensor input, int dim, long start, long length) {
        int normalized = input.shape().normalizeDim(dim);
        long size = input.size(normalized);
        if (start < 0 || length < 0 || start + length > size) {
            throw new IndexOutOfBoundsException(
                    "start (" + start + ") + length (" + length + ") exceeds dimension size (" + size + ")");
        }
        long offset = input.storageOffset() + start * input.stride(normalized);
        long[] sizes = input.shape().toArray();
        sizes[normalized] = length;
        return new ViewSpec(Shape.of(sizes), input.strides(), offset, false);
    }

    static ViewSpec select(Tensor input, int dim, long index) {
        int normalized = input.shape().normalizeDim(dim);
        long size = input.size(normalized);
        if (index < -size || index >= size) {
            throw new IndexOutOfBoundsException("index " + index + " is out of bounds for dimension " + dim
                    + " with size " + size);
        }
        long actual = index < 0 ? index + size : index;
        long offset = input.storageOffset() + actual * input.stride(normalized);
        long[] sizes = new long[input.dim() - 1];
        long[] strides = new long[input.dim() - 1];
        int target = 0;
        for (int i = 0; i < input.dim(); i++) {
            if (i == normalized) {
                continue;
            }
            sizes[target] = input.size(i);
            strides[target] = input.stride(i);
            target++;
        }
        return new ViewSpec(Shape.of(sizes), Strides.of(strides), offset, false);
    }

    static ViewSpec squeeze(Tensor input) {
        long[] sizes = new long[input.dim()];
        long[] strides = new long[input.dim()];
        int target = 0;
        for (int i = 0; i < input.dim(); i++) {
            if (input.size(i) == 1) {
                continue;
            }
            sizes[target] = input.size(i);
            strides[target] = input.stride(i);
            target++;
        }
        return new ViewSpec(
                Shape.of(Arrays.copyOf(sizes, target)),
                Strides.of(Arrays.copyOf(strides, target)),
                input.storageOffset(),
                false);
    }

    static ViewSpec squeezeDim(Tensor input, int dim) {
        if (input.dim() == 0) {
            if (dim != 0 && dim != -1) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            return new ViewSpec(input.shape(), input.strides(), input.storageOffset(), false);
        }
        int normalized = input.shape().normalizeDim(dim);
        if (input.size(normalized) != 1) {
            return new ViewSpec(input.shape(), input.strides(), input.storageOffset(), false);
        }
        long[] sizes = new long[input.dim() - 1];
        long[] strides = new long[input.dim() - 1];
        int target = 0;
        for (int i = 0; i < input.dim(); i++) {
            if (i == normalized) {
                continue;
            }
            sizes[target] = input.size(i);
            strides[target] = input.stride(i);
            target++;
        }
        return new ViewSpec(Shape.of(sizes), Strides.of(strides), input.storageOffset(), false);
    }

    static ViewSpec unsqueeze(Tensor input, int dim) {
        int rank = input.dim();
        int normalized = dim < 0 ? dim + rank + 1 : dim;
        if (normalized < 0 || normalized > rank) {
            throw new IndexOutOfBoundsException("Dimension out of range (expected to be in range of [" + (-rank - 1)
                    + ", " + rank + "], but got " + dim + ")");
        }
        long[] sizes = new long[rank + 1];
        long[] strides = new long[rank + 1];
        for (int i = 0; i < normalized; i++) {
            sizes[i] = input.size(i);
            strides[i] = input.stride(i);
        }
        sizes[normalized] = 1;
        strides[normalized] = normalized < rank ? input.size(normalized) * input.stride(normalized) : 1;
        for (int i = normalized; i < rank; i++) {
            sizes[i + 1] = input.size(i);
            strides[i + 1] = input.stride(i);
        }
        return new ViewSpec(Shape.of(sizes), Strides.of(strides), input.storageOffset(), false);
    }

    static ViewSpec expand(Tensor input, long[] requested) {
        if (requested.length < input.dim()) {
            throw new ShapeException("expand: the number of sizes provided (" + requested.length
                    + ") must be at least the number of dimensions (" + input.dim() + ")");
        }
        int extra = requested.length - input.dim();
        long[] sizes = new long[requested.length];
        long[] strides = new long[requested.length];
        for (int i = 0; i < extra; i++) {
            if (requested[i] < 0) {
                throw new ShapeException("expand: the expanded size of the tensor (-1) isn't allowed in a leading "
                        + "non-existing dimension");
            }
            sizes[i] = requested[i];
            strides[i] = 0;
        }
        for (int i = 0; i < input.dim(); i++) {
            long requestedSize = requested[extra + i];
            long current = input.size(i);
            if (requestedSize == -1) {
                requestedSize = current;
            }
            if (current == requestedSize) {
                sizes[extra + i] = current;
                strides[extra + i] = input.stride(i);
            } else if (current == 1) {
                sizes[extra + i] = requestedSize;
                strides[extra + i] = 0;
            } else {
                throw new ShapeException("The expanded size of the tensor (" + requestedSize
                        + ") must match the existing size (" + current + ") at non-singleton dimension " + (extra + i));
            }
        }
        return new ViewSpec(Shape.of(sizes), Strides.of(strides), input.storageOffset(), false);
    }

    /** Builds a CPU view with the given geometry, sharing storage and the version counter. */
    static Tensor cpuView(Tensor input, ViewSpec spec, boolean requiresGrad) {
        return Tensor.wrap(new TensorImpl(
                input.impl().storage(),
                spec.offset(),
                spec.shape(),
                spec.strides(),
                input.versionCounter(),
                requiresGrad,
                true));
    }

    /** Builds a meta view; meta tensors have no storage but preserve shape/stride metadata. */
    static Tensor metaView(Tensor input, ViewSpec spec, boolean requiresGrad) {
        return Tensor.wrap(TensorImpl.meta(spec.shape(), spec.strides(), input.dtype(), requiresGrad));
    }

    /** Copies {@code input} into a fresh contiguous CPU tensor with the given shape. */
    static Tensor contiguousCopy(Tensor input, Shape shape, boolean requiresGrad) {
        Tensor output = Tensor.wrap(TensorImpl.fresh(
                org.jtorch.core.CpuAllocator.allocate(input.dtype(), shape.numel()),
                shape,
                Strides.contiguous(shape),
                requiresGrad));
        Copy.copy(input, output);
        return output;
    }

    /** Reinterprets a contiguous tensor as another shape sharing storage. */
    static Tensor contiguousReshape(Tensor input, Shape shape) {
        if (!input.isContiguous()) {
            throw new ShapeException("contiguousReshape requires a contiguous input");
        }
        return cpuView(input, new ViewSpec(shape, Strides.contiguous(shape), input.storageOffset(), false),
                input.requiresGrad());
    }

    /** Creates a fresh CPU tensor with explicit strides and the requested shape. */
    static Tensor empty(Tensor input, Shape shape, Strides strides, boolean requiresGrad) {
        return empty(input.dtype(), shape, strides, requiresGrad);
    }

    static Tensor empty(org.jtorch.core.DType dtype, Shape shape, Strides strides, boolean requiresGrad) {
        long storageElements = requiredElements(shape, strides);
        return Tensor.wrap(TensorImpl.fresh(
                org.jtorch.core.CpuAllocator.allocate(dtype, storageElements), shape, strides, requiresGrad));
    }

    static long requiredElements(Shape shape, Strides strides) {
        long max = 0;
        for (int i = 0; i < shape.rank(); i++) {
            if (shape.dim(i) == 0) {
                return 0;
            }
            max += (shape.dim(i) - 1) * strides.stride(i);
        }
        return max + 1;
    }

    static Scalar requireScalar(org.jtorch.core.value.Value[] args, int index, String op) {
        org.jtorch.core.value.Value value = args[index];
        if (value.isNone()) {
            throw new IllegalArgumentException(op + " requires argument " + index);
        }
        return value.asScalar();
    }

    static Device deviceOf(org.jtorch.core.value.Value[] args, int index) {
        return args[index].asDevice();
    }
}
