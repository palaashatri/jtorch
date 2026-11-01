package org.jtorch.core;

import java.util.ServiceLoader;
import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.spi.AutogradProvider;
import org.jtorch.core.spi.BackendProvider;
import org.jtorch.core.spi.GradientEngine;

/**
 * Runtime bootstrap. Backends and the autograd engine are discovered with {@link ServiceLoader} from the module
 * path (JPMS {@code provides}) or the class path ({@code META-INF/services}).
 */
public final class JTorch {
    private static final Object LOCK = new Object();
    private static volatile boolean initialized;
    private static volatile GradientEngine gradientEngine;

    private JTorch() {}

    public static void ensureInitialized() {
        if (initialized) {
            return;
        }
        synchronized (LOCK) {
            if (initialized) {
                return;
            }
            Dispatcher dispatcher = Dispatcher.get();
            for (BackendProvider provider : ServiceLoader.load(BackendProvider.class)) {
                provider.register(dispatcher);
            }
            for (AutogradProvider provider : ServiceLoader.load(AutogradProvider.class)) {
                provider.register(dispatcher);
                gradientEngine = provider.engine();
            }
            initialized = true;
        }
    }

    public static GradientEngine gradientEngine() {
        ensureInitialized();
        GradientEngine engine = gradientEngine;
        if (engine == null) {
            throw new AutogradException(
                    "No autograd provider is available; add the jtorch-autograd module to the module path");
        }
        return engine;
    }

    /** True once providers have been loaded; used by diagnostics. */
    public static boolean isInitialized() {
        return initialized;
    }
}
