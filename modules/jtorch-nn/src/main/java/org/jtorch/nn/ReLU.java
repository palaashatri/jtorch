package org.jtorch.nn;

import org.jtorch.core.Tensor;

/** Elementwise rectified linear unit module. */
public final class ReLU extends Module {
    public ReLU() {}

    @Override
    public Tensor forward(Tensor input) {
        return input.relu();
    }
}
