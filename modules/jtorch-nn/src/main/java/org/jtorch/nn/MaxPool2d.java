package org.jtorch.nn;

import org.jtorch.core.Tensor;

/** 2-D max pooling over NCHW tensors. */
public final class MaxPool2d extends Module {
    private final int[] kernel;
    private final int[] stride;
    private final int[] padding;
    private final int[] dilation;
    private final boolean ceilMode;

    public MaxPool2d(int kernel, int stride, int padding) {
        this(new int[] {kernel, kernel}, new int[] {stride, stride}, new int[] {padding, padding},
                new int[] {1, 1}, false);
    }

    public MaxPool2d(int[] kernel, int[] stride, int[] padding, int[] dilation, boolean ceilMode) {
        if (kernel.length != 2 || stride.length != 2 || padding.length != 2 || dilation.length != 2) {
            throw new IllegalArgumentException("MaxPool2d arguments need two entries");
        }
        this.kernel = kernel.clone();
        this.stride = stride.clone();
        this.padding = padding.clone();
        this.dilation = dilation.clone();
        this.ceilMode = ceilMode;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.maxPool2d(kernel, stride, padding, dilation, ceilMode);
    }
}
