package org.jtorch.nn;

import org.jtorch.core.Tensor;

/** Elementwise hyperbolic tangent module. */
public final class Tanh extends Module {
    public Tanh() {}

    @Override
    public Tensor forward(Tensor input) {
        return input.tanh();
    }
}
