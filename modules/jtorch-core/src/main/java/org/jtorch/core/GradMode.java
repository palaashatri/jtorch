package org.jtorch.core;

import java.util.function.Supplier;

/**
 * Dynamically scoped grad and inference mode using finalized {@link ScopedValue}. There is no inheritable thread
 * local state: numerical work spawned on worker threads must carry an explicit mode decision.
 */
public final class GradMode {
    private static final ScopedValue<Boolean> GRAD_ENABLED = ScopedValue.newInstance();
    private static final ScopedValue<Boolean> INFERENCE = ScopedValue.newInstance();

    private GradMode() {}

    public static boolean isEnabled() {
        return GRAD_ENABLED.orElse(true) && !isInferenceMode();
    }

    public static boolean isInferenceMode() {
        return INFERENCE.orElse(false);
    }

    public static void noGrad(Runnable action) {
        ScopedValue.where(GRAD_ENABLED, false).run(action);
    }

    public static <T> T noGrad(Supplier<T> action) {
        return ScopedValue.where(GRAD_ENABLED, false).call(action::get);
    }

    public static void inferenceMode(Runnable action) {
        ScopedValue.where(INFERENCE, true).run(action);
    }

    public static <T> T inferenceMode(Supplier<T> action) {
        return ScopedValue.where(INFERENCE, true).call(action::get);
    }

    public static void enableGrad(Runnable action) {
        ScopedValue.where(GRAD_ENABLED, true).run(action);
    }

    public static <T> T enableGrad(Supplier<T> action) {
        return ScopedValue.where(GRAD_ENABLED, true).call(action::get);
    }
}
