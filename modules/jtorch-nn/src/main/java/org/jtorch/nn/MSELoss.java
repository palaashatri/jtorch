package org.jtorch.nn;

import org.jtorch.core.Tensor;
import org.jtorch.nn.functional.F;

/**
 * Mean/sum/none squared-error criterion. Loss objects are parameterless and deliberately not modules: they hold no
 * state and are invoked with an explicit target.
 */
public final class MSELoss {
    private final F.Reduction reduction;

    public MSELoss() {
        this(F.Reduction.MEAN);
    }

    public MSELoss(F.Reduction reduction) {
        this.reduction = reduction;
    }

    public Tensor forward(Tensor input, Tensor target) {
        return F.mseLoss(input, target, reduction);
    }

    public Tensor call(Tensor input, Tensor target) {
        return forward(input, target);
    }

    public F.Reduction reduction() {
        return reduction;
    }
}
