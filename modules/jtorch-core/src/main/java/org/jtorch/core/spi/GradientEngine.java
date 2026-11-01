package org.jtorch.core.spi;

import org.jtorch.core.Tensor;

/** Reverse-mode execution entry point; implemented by the autograd module. */
public interface GradientEngine {
    /**
     * Runs reverse-mode differentiation from {@code root}.
     *
     * @param root tensor to differentiate
     * @param gradient explicit seed gradient; {@code null} means a scalar-seeded backward
     * @param retainGraph keep the graph alive for another backward call
     * @param createGraph build a graph while differentiating (higher-order gradients); not supported yet
     */
    void backward(Tensor root, Tensor gradient, boolean retainGraph, boolean createGraph);
}
