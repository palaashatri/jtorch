package org.jtorch.nn;

import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.nn.functional.F;

/** Affine transform {@code y = x W^T + b}. */
public final class Linear extends Module {
    private final Parameter weight;
    private final Parameter bias;
    private final int inFeatures;
    private final int outFeatures;

    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, true);
    }

    public Linear(int inFeatures, int outFeatures, boolean useBias) {
        if (inFeatures <= 0 || outFeatures <= 0) {
            throw new IllegalArgumentException("Linear features must be positive");
        }
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weight = registerParameter("weight", new Parameter(Init.uniform(Shape.of(outFeatures, inFeatures),
                inFeatures)));
        this.bias = useBias
                ? registerParameter("bias", new Parameter(Init.uniform(Shape.of(outFeatures), inFeatures)))
                : null;
    }

    @Override
    public Tensor forward(Tensor input) {
        return F.linear(input, weight, bias);
    }

    public Parameter weight() {
        return weight;
    }

    public Parameter bias() {
        return bias;
    }

    public int inFeatures() {
        return inFeatures;
    }

    public int outFeatures() {
        return outFeatures;
    }
}
