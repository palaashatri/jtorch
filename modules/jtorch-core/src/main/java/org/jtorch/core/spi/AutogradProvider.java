package org.jtorch.core.spi;

import org.jtorch.core.Tensor;
import org.jtorch.core.dispatch.Dispatcher;

/** Service-provider interface implemented by the autograd module. */
public interface AutogradProvider {
    /** Registers autograd kernels for every differentiable operator. */
    void register(Dispatcher dispatcher);

    GradientEngine engine();
}
