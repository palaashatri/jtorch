package org.jtorch.vision;

import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.nn.Module;

/** Gradient-based saliency for a single class score. */
public final class Saliency {
    private Saliency() {}

    public static Tensor inputGradient(Module model, Tensor input, int classIndex) {
        if (input.dim() != 4 || input.size(0) != 1) {
            throw new IllegalArgumentException("Saliency expects a [1, C, H, W] input");
        }
        Tensor tracked = input.detach().requiresGrad(true);
        Tensor logits = model.call(tracked);
        if (logits.dim() != 2 || logits.size(0) != 1) {
            throw new IllegalArgumentException("Saliency requires a model that returns [1, classes] logits");
        }
        if (classIndex < 0 || classIndex >= logits.size(1)) {
            throw new IllegalArgumentException("classIndex is out of range");
        }
        logits.select(1, classIndex).sum().backward();
        return tracked.grad().detach();
    }

    /** Absolute gradient magnitude reduced over channels, normalized to [0, 1]. */
    public static Tensor heatmap(Tensor gradient) {
        Tensor magnitude = gradient.abs().maxWithIndices(1, false).values();
        double minimum = magnitude.min().item();
        double maximum = magnitude.max().item();
        if (maximum - minimum < 1e-12) {
            return magnitude.mul(0.0);
        }
        return magnitude.sub(minimum).div(maximum - minimum);
    }
}
