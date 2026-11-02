package org.jtorch.optim;

/** Base class for learning-rate schedules. */
public abstract class LRScheduler {
    protected final Optimizer optimizer;
    protected final double baseLearningRate;
    protected int stepCount;

    protected LRScheduler(Optimizer optimizer) {
        this.optimizer = optimizer;
        this.baseLearningRate = optimizer.learningRate();
    }

    /** Advances the schedule by one step and applies the new learning rate to the optimizer. */
    public double step() {
        stepCount++;
        double learningRate = compute();
        optimizer.setLearningRate(learningRate);
        return learningRate;
    }

    protected abstract double compute();

    public int stepCount() {
        return stepCount;
    }

    public double lastLearningRate() {
        return optimizer.learningRate();
    }
}
