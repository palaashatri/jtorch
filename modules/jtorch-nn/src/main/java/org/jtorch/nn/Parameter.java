package org.jtorch.nn;

import org.jtorch.core.Tensor;
import org.jtorch.core.TensorImpl;

/**
 * A trainable tensor with stable module identity.
 *
 * <p>A parameter wraps its source tensor handle directly rather than creating a second autograd metadata object. This
 * matters for the documented vertical slice, where the optimizer is constructed from an existing tensor after
 * {@code backward()}: a separate handle would not see the source gradient and {@code step()} would silently do
 * nothing. Non-leaf sources fall back to a fresh view because their autograd metadata cannot be re-tagged.
 */
public class Parameter extends Tensor {
    public Parameter(Tensor data) {
        this(data, true);
    }

    public Parameter(Tensor data, boolean requiresGrad) {
        super(wrap(data, requiresGrad));
    }

    private static TensorImpl wrap(Tensor data, boolean requiresGrad) {
        if (data.device().isMeta()) {
            throw new IllegalArgumentException("Parameters require materialised storage; meta tensors are not supported");
        }
        TensorImpl impl = data.impl();
        if (impl.gradFn() == null) {
            impl.setRequiresGrad(requiresGrad);
            return impl;
        }
        return new TensorImpl(
                impl.storage(),
                data.storageOffset(),
                data.shape(),
                data.strides(),
                data.versionCounter(),
                requiresGrad,
                true);
    }

    public Parameter detachCopy() {
        return new Parameter(detach().clone());
    }
}
