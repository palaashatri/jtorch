package org.jtorch.core.dispatch;

import org.jtorch.core.Op;
import org.jtorch.core.value.Value;

/**
 * A backend implementation of one operator for one dispatch key.
 *
 * <p>Kernels receive the canonical argument list and the dispatch key set that selected them. Wrapper kernels
 * (autograd, autocast) must redispatch through {@link Dispatcher#callWithKeys} after removing their own key; they
 * may never call a backend kernel directly.
 */
@FunctionalInterface
public interface Kernel {
    Value[] invoke(Op op, Value[] args, DispatchKeySet keys);
}
