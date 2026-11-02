package org.jtorch.optim;

import org.jtorch.core.Shape;
import org.jtorch.core.Scalar;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;

/** Scalar constants in a tensor's dtype/device, used by update formulas. */
final class OptimUtil {
    private OptimUtil() {}

    static Tensor scalarLike(Tensor reference, double value) {
        return org.jtorch.api.Torch.full(
                Shape.scalar(),
                Scalar.of(value),
                TensorOptions.defaults().dtype(reference.dtype()).device(reference.device()));
    }
}
