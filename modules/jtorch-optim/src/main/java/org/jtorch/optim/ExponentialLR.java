package org.jtorch.optim;

/** Multiplies the learning rate by {@code gamma} on every scheduler step. */
public final class ExponentialLR extends LRScheduler {
    private final double gamma;

    public ExponentialLR(Optimizer optimizer, double gamma) {
        super(optimizer);
        this.gamma = gamma;
    }

    @Override
    protected double compute() {
        return baseLearningRate * Math.pow(gamma, stepCount);
    }
}
