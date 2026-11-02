package org.jtorch.cpu;

import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.memory.TensorAccess;

/** Broadcast-aware copy used by view copies and in-place {@code copy_}. */
final class Copy {
    private Copy() {}

    /** Copies between tensors of identical shape, honouring both tensors' strides. */
    static void copy(Tensor source, Tensor target) {
        if (!source.shape().equals(target.shape())) {
            throw new org.jtorch.core.ShapeException(
                    "copy shape mismatch: " + source.shape() + " vs " + target.shape());
        }
        Iteration.forEach(target, new Tensor[] {source}, (outOffset, inOffsets) ->
                copyElement(source, target, inOffsets[0], outOffset));
    }

    /** Copies a broadcastable source into a target with the target's full shape. */
    static void copyBroadcast(Tensor source, Tensor target) {
        Shape targetShape = target.shape();
        if (source.dim() > targetShape.rank()) {
            throw new org.jtorch.core.ShapeException(
                    "copy_: source rank " + source.dim() + " exceeds target rank " + targetShape.rank());
        }
        int offset = targetShape.rank() - source.dim();
        for (int i = 0; i < source.dim(); i++) {
            long sourceSize = source.size(i);
            long targetSize = targetShape.dim(offset + i);
            if (sourceSize != 1 && sourceSize != targetSize) {
                throw new org.jtorch.core.ShapeException("copy_: source shape " + source.shape()
                        + " cannot broadcast to target " + targetShape);
            }
        }
        Iteration.forEach(target, new Tensor[] {source}, (outOffset, inOffsets) ->
                copyElement(source, target, inOffsets[0], outOffset));
    }

    private static void copyElement(Tensor source, Tensor target, long sourceOffset, long targetOffset) {
        if (target.dtype().isBoolean()) {
            TensorAccess.setBoolean(target.impl(), targetOffset,
                    TensorAccess.getBoolean(source.impl(), sourceOffset));
        } else if (target.dtype().isIntegral()) {
            TensorAccess.setLong(target.impl(), targetOffset, TensorAccess.getLong(source.impl(), sourceOffset));
        } else {
            TensorAccess.setDouble(target.impl(), targetOffset, TensorAccess.getDouble(source.impl(), sourceOffset));
        }
    }
}
