package org.jtorch.optim;

/** Decays the learning rate by {@code gamma} every {@code stepSize} scheduler steps. */
public final class StepLR extends LRScheduler {
    private final int stepSize;
    private final double gamma;

    public StepLR(Optimizer optimizer, int stepSize, double gamma) {
        super(optimizer);
        if (stepSize <= 0) {
            throw new IllegalArgumentException("stepSize must be positive");
        }
        this.stepSize = stepSize;
        this.gamma = gamma;
    }

    @Override
    protected double compute() {
        int exponent = stepCount / stepSize;
        return baseLearningRate * Math.pow(gamma, exponent);
    }
}
