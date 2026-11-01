package org.jtorch.autograd;

import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.spi.AutogradProvider;
import org.jtorch.core.spi.GradientEngine;

/** Registers native backward formulas and supplies the reverse-mode engine. */
public final class AutogradProviderImpl implements AutogradProvider {
    private final AutogradEngine engine = new AutogradEngine();

    public AutogradProviderImpl() {}

    @Override
    public void register(Dispatcher dispatcher) {
        BackwardKernels.register(dispatcher);
    }

    @Override
    public GradientEngine engine() {
        return engine;
    }
}
