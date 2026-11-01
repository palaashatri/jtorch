package org.jtorch.core.spi;

import org.jtorch.core.dispatch.Dispatcher;

/** Service-provider interface implemented by backend modules such as the CPU reference backend. */
public interface BackendProvider {
    void register(Dispatcher dispatcher);
}
