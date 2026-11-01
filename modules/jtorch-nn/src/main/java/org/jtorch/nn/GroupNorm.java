package org.jtorch.nn;

import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;

/** Group normalization over NCHW tensors. */
public final class GroupNorm extends Module {
    private final Parameter weight;
    private final Parameter bias;
    private final int groups;
    private final int channels;
    private final double eps;

    public GroupNorm(int groups, int channels) {
        this(groups, channels, 1e-5);
    }

    public GroupNorm(int groups, int channels, double eps) {
        if (groups <= 0 || channels <= 0 || channels % groups != 0) {
            throw new IllegalArgumentException("GroupNorm requires positive channels divisible by groups");
        }
        this.groups = groups;
        this.channels = channels;
        this.eps = eps;
        this.weight = registerParameter("weight", new Parameter(Torch.ones(Shape.of(channels))));
        this.bias = registerParameter("bias", new Parameter(Torch.zeros(Shape.of(channels))));
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.groupNorm(groups, weight, bias, eps);
    }

    public Parameter weight() {
        return weight;
    }

    public Parameter bias() {
        return bias;
    }
}
