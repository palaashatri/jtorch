package org.jtorch.nn;

import org.jtorch.core.Tensor;

/** 2-D average pooling over NCHW tensors. */
public final class AvgPool2d extends Module {
    private final int[] kernel;
    private final int[] stride;
    private final int[] padding;
    private final boolean ceilMode;
    private final boolean countIncludePad;

    public AvgPool2d(int kernel, int stride, int padding) {
        this(new int[] {kernel, kernel}, new int[] {stride, stride}, new int[] {padding, padding}, false, true);
    }

    public AvgPool2d(int[] kernel, int[] stride, int[] padding, boolean ceilMode, boolean countIncludePad) {
        if (kernel.length != 2 || stride.length != 2 || padding.length != 2) {
            throw new IllegalArgumentException("AvgPool2d arguments need two entries");
        }
        for (int i = 0; i < 2; i++) {
            if (kernel[i] <= 0 || stride[i] <= 0 || padding[i] < 0) {
                throw new IllegalArgumentException("AvgPool2d kernel/stride must be positive and padding non-negative");
            }
        }
        if (ceilMode) {
            throw new IllegalArgumentException("AvgPool2d in this build does not support ceilMode=true");
        }
        this.kernel = kernel.clone();
        this.stride = stride.clone();
        this.padding = padding.clone();
        this.ceilMode = false;
        this.countIncludePad = countIncludePad;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.avgPool2d(kernel, stride, padding, ceilMode, countIncludePad);
    }
}
