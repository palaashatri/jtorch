package org.jtorch.cpu;

import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.spi.BackendProvider;

/** Provides the CPU reference and meta kernels to the dispatcher. */
public final class CpuBackendProvider implements BackendProvider {
    public CpuBackendProvider() {}

    @Override
    public void register(Dispatcher dispatcher) {
        MetaKernels.register(dispatcher);
        CpuKernels.register(dispatcher);
        CompositeKernels.register(dispatcher);
    }
}
