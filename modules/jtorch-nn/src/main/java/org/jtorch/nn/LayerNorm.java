package org.jtorch.nn;

import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;

/** Layer normalization over trailing dimensions. */
public final class LayerNorm extends Module {
    private final Parameter weight;
    private final Parameter bias;
    private final long[] normalizedShape;
    private final double eps;

    public LayerNorm(long... normalizedShape) {
        this(1e-5, normalizedShape);
    }

    public LayerNorm(double eps, long... normalizedShape) {
        if (normalizedShape.length == 0) {
            throw new IllegalArgumentException("LayerNorm requires at least one normalized dimension");
        }
        this.normalizedShape = normalizedShape.clone();
        this.eps = eps;
        this.weight = registerParameter("weight", new Parameter(Torch.ones(Shape.of(normalizedShape))));
        this.bias = registerParameter("bias", new Parameter(Torch.zeros(Shape.of(normalizedShape))));
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.layerNorm(normalizedShape, weight, bias, eps);
    }

    public Parameter weight() {
        return weight;
    }

    public Parameter bias() {
        return bias;
    }
}
