package org.jtorch.cpu;

import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;

/** Odometer iteration over a logical output shape with right-aligned broadcast input offsets. */
final class Iteration {
    private Iteration() {}

    @FunctionalInterface
    interface Body {
        void accept(long outputOffset, long[] inputOffsets);
    }

    static void forEach(Tensor output, Tensor[] inputs, Body body) {
        Shape shape = output.shape();
        int rank = shape.rank();
        long numel = shape.numel();
        if (numel == 0) {
            return;
        }
        long[] outputStrides = output.strides().toArray();
        long[][] inputStrides = new long[inputs.length][];
        long[] inputOffsets = new long[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            inputStrides[i] = ShapeRules.broadcastStrides(inputs[i].impl(), shape);
            inputOffsets[i] = inputs[i].storageOffset();
        }
        long[] index = new long[rank];
        long outputOffset = output.storageOffset();
        while (true) {
            body.accept(outputOffset, inputOffsets);
            int dim = rank - 1;
            while (dim >= 0) {
                index[dim]++;
                outputOffset += outputStrides[dim];
                for (int i = 0; i < inputs.length; i++) {
                    inputOffsets[i] += inputStrides[i][dim];
                }
                if (index[dim] < shape.dim(dim)) {
                    break;
                }
                outputOffset -= outputStrides[dim] * index[dim];
                for (int i = 0; i < inputs.length; i++) {
                    inputOffsets[i] -= inputStrides[i][dim] * index[dim];
                }
                index[dim] = 0;
                dim--;
            }
            if (dim < 0) {
                return;
            }
        }
    }
}
