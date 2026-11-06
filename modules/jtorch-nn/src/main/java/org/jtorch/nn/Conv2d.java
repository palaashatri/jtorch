package org.jtorch.nn;

import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;

/** 2-D convolution over NCHW tensors with PyTorch-compatible parameter layout. */
public final class Conv2d extends Module {
    private final Parameter weight;
    private final Parameter bias;
    private final int inChannels;
    private final int outChannels;
    private final int[] kernel;
    private final int[] stride;
    private final int[] padding;
    private final int[] dilation;
    private final int groups;

    public Conv2d(int inChannels, int outChannels, int kernel) {
        this(inChannels, outChannels, new int[] {kernel, kernel}, new int[] {1, 1}, new int[] {0, 0},
                new int[] {1, 1}, 1, true);
    }

    public Conv2d(int inChannels, int outChannels, int[] kernel, int[] stride, int[] padding) {
        this(inChannels, outChannels, kernel, stride, padding, new int[] {1, 1}, 1, true);
    }

    public Conv2d(
            int inChannels,
            int outChannels,
            int[] kernel,
            int[] stride,
            int[] padding,
            int[] dilation,
            int groups,
            boolean useBias) {
        if (groups <= 0 || inChannels % groups != 0 || outChannels % groups != 0) {
            throw new IllegalArgumentException("Invalid Conv2d channel/groups configuration");
        }
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("Conv2d channels must be positive");
        }
        if (kernel.length != 2 || stride.length != 2 || padding.length != 2 || dilation.length != 2) {
            throw new IllegalArgumentException("Conv2d kernel/stride/padding/dilation need two entries");
        }
        for (int i = 0; i < 2; i++) {
            if (kernel[i] <= 0 || stride[i] <= 0 || dilation[i] <= 0 || padding[i] < 0) {
                throw new IllegalArgumentException(
                        "Conv2d kernel and stride/dilation must be positive and padding non-negative");
            }
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernel = kernel.clone();
        this.stride = stride.clone();
        this.padding = padding.clone();
        this.dilation = dilation.clone();
        this.groups = groups;
        double fanIn = (double) inChannels / groups * kernel[0] * kernel[1];
        this.weight = registerParameter("weight",
                new Parameter(Init.uniform(Shape.of(outChannels, inChannels / groups, kernel[0], kernel[1]), fanIn)));
        this.bias = useBias
                ? registerParameter("bias", new Parameter(Init.uniform(Shape.of(outChannels), fanIn)))
                : null;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.conv2d(weight, bias, stride, padding, dilation, groups);
    }

    public Parameter weight() {
        return weight;
    }

    public Parameter bias() {
        return bias;
    }

    public int inChannels() {
        return inChannels;
    }

    public int outChannels() {
        return outChannels;
    }

    public int groups() {
        return groups;
    }
}
