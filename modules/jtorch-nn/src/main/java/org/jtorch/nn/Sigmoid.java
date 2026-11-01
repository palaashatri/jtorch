package org.jtorch.nn;

import org.jtorch.core.Tensor;

/** Elementwise logistic sigmoid module. */
public final class Sigmoid extends Module {
    public Sigmoid() {}

    @Override
    public Tensor forward(Tensor input) {
        return input.sigmoid();
    }
}
