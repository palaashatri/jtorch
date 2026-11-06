package org.jtorch.nn;

import org.jtorch.core.Tensor;

/** Adaptive average pooling to a fixed output size over CHW or NCHW tensors. */
public final class AdaptiveAvgPool2d extends Module {
    private final int[] outputSize;

    public AdaptiveAvgPool2d(int outputSize) {
        this(new int[] {outputSize, outputSize});
    }

    public AdaptiveAvgPool2d(int[] outputSize) {
        if (outputSize.length != 1 && outputSize.length != 2) {
            throw new IllegalArgumentException("AdaptiveAvgPool2d output size needs one or two entries");
        }
        for (int value : outputSize) {
            if (value < 0) {
                throw new IllegalArgumentException("AdaptiveAvgPool2d output size must be non-negative");
            }
        }
        this.outputSize = outputSize.clone();
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.adaptiveAvgPool2d(outputSize);
    }
}
