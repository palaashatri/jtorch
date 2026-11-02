package org.jtorch.nn;

import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.jtorch.core.memory.TensorAccess;

/** Initialisation helpers matching the distributions PyTorch uses for built-in layers. */
final class Init {
    private Init() {}

    /** PyTorch Linear.reset_parameters: U(-1/sqrt(fan_in), 1/sqrt(fan_in)). */
    static Tensor uniform(Shape shape, double fanIn) {
        double bound = 1.0 / Math.sqrt(fanIn);
        Tensor random = Torch.rand(shape);
        Tensor scale = Torch.full(Shape.scalar(), org.jtorch.core.Scalar.of(2 * bound));
        Tensor shift = Torch.full(Shape.scalar(), org.jtorch.core.Scalar.of(bound));
        return random.mul(scale).sub(shift);
    }

    static Tensor zeros(Shape shape) {
        return Torch.zeros(shape, TensorOptions.defaults().dtype(DType.FLOAT32));
    }

    static void fill(Tensor tensor, double value) {
        for (long i = 0; i < tensor.numel(); i++) {
            TensorAccess.setDouble(tensor.impl(), tensor.elementOffset(i), value);
        }
    }
}
