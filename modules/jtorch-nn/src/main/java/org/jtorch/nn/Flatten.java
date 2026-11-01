package org.jtorch.nn;

import org.jtorch.core.Tensor;

/** Flattens a contiguous range of dimensions; defaults follow PyTorch ({@code start_dim=1}). */
public final class Flatten extends Module {
    private final int startDim;
    private final int endDim;

    public Flatten() {
        this(1, -1);
    }

    public Flatten(int startDim, int endDim) {
        this.startDim = startDim;
        this.endDim = endDim;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.flatten(startDim, endDim);
    }
}
