package org.jtorch.optim;

import java.util.List;
import org.jtorch.nn.Parameter;

/** Base optimizer contract. */
public abstract class Optimizer {
    protected final List<Parameter> parameters;
    private double learningRate;

    protected Optimizer(List<Parameter> parameters, double learningRate) {
        if (parameters.isEmpty()) {
            throw new IllegalArgumentException("Optimizer requires at least one parameter");
        }
        if (learningRate <= 0 || !Double.isFinite(learningRate)) {
            throw new IllegalArgumentException("learningRate must be positive and finite, got " + learningRate);
        }
        this.parameters = List.copyOf(parameters);
        this.learningRate = learningRate;
    }

    public final List<Parameter> parameters() {
        return parameters;
    }

    public final double learningRate() {
        return learningRate;
    }

    public final void setLearningRate(double value) {
        if (value < 0 || !Double.isFinite(value)) {
            throw new IllegalArgumentException("learningRate must be non-negative and finite");
        }
        this.learningRate = value;
    }

    /** Applies one parameter update using the currently accumulated gradients. */
    public abstract void step();

    /** Clears accumulated gradients (PyTorch's default {@code set_to_none=True}). */
    public void zeroGrad() {
        for (Parameter parameter : parameters) {
            parameter.impl().setGrad(null);
        }
    }
}
