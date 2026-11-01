package org.jtorch.core.dispatch;

/**
 * Dispatch keys ordered by priority. Higher ordinal means higher priority; the dispatcher consults keys from the
 * top of the set down.
 */
public enum DispatchKey {
    /** Functionalisation intercepts mutations for transforms and compilation (reserved for M5). */
    FUNCTIONALIZE(600),
    /** Autocast intercepts floating point operators (reserved for M9). */
    AUTOCAST_CPU(500),
    /** Reverse-mode autograd wrapping of differentiable operators. */
    AUTOGRAD(400),
    /** Shape-only execution for fake/meta tensors. */
    META(300),
    /** The CPU reference backend. */
    CPU(200),
    /** Backend-independent decompositions expressed with primitive operators. */
    COMPOSITE(100);

    private final int priority;

    DispatchKey(int priority) {
        this.priority = priority;
    }

    public int priority() {
        return priority;
    }
}
