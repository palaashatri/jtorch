package org.jtorch.nn;

import org.jtorch.core.Tensor;
import org.jtorch.nn.functional.F;

/** Softmax cross-entropy criterion over int64 class targets. */
public final class CrossEntropyLoss {
    private final F.Reduction reduction;

    public CrossEntropyLoss() {
        this(F.Reduction.MEAN);
    }

    public CrossEntropyLoss(F.Reduction reduction) {
        this.reduction = reduction;
    }

    public Tensor forward(Tensor logits, Tensor target) {
        return F.crossEntropy(logits, target, reduction);
    }

    public Tensor call(Tensor logits, Tensor target) {
        return forward(logits, target);
    }

    public F.Reduction reduction() {
        return reduction;
    }
}
