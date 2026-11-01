package org.jtorch.optim;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import org.jtorch.api.Torch;
import org.jtorch.core.Tensor;
import org.jtorch.nn.Parameter;

/**
 * Stochastic gradient descent with optional momentum, weight decay, dampening and Nesterov momentum. The update
 * matches {@code torch.optim.SGD}.
 */
public final class SGD extends Optimizer {
    private final double momentum;
    private final double weightDecay;
    private final double dampening;
    private final boolean nesterov;
    private final Map<Parameter, Tensor> momentumBuffers = new IdentityHashMap<>();

    private SGD(Builder builder) {
        super(builder.parameters, builder.learningRate);
        this.momentum = builder.momentum;
        this.weightDecay = builder.weightDecay;
        this.dampening = builder.dampening;
        this.nesterov = builder.nesterov;
        if (momentum < 0) {
            throw new IllegalArgumentException("momentum must be non-negative");
        }
        if (weightDecay < 0) {
            throw new IllegalArgumentException("weightDecay must be non-negative");
        }
        if (dampening < 0) {
            throw new IllegalArgumentException("dampening must be non-negative");
        }
        if (nesterov && (momentum <= 0 || dampening != 0)) {
            throw new IllegalArgumentException("Nesterov momentum requires momentum > 0 and dampening == 0");
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public void step() {
        Torch.noGrad(() -> {
            for (Parameter parameter : parameters) {
                Tensor gradient = parameter.grad();
                if (gradient == null) {
                    continue;
                }
                if (weightDecay != 0) {
                    gradient = gradient.add(parameter, weightDecay);
                }
                Tensor update = gradient;
                if (momentum != 0) {
                    Tensor buffer = momentumBuffers.get(parameter);
                    if (buffer == null) {
                        buffer = gradient.clone();
                        momentumBuffers.put(parameter, buffer);
                    } else {
                        buffer.mul_(OptimUtil.scalarLike(buffer, momentum)).add_(gradient, 1 - dampening);
                    }
                    update = nesterov ? gradient.add(buffer, momentum) : buffer;
                }
                parameter.sub_(update, learningRate());
            }
        });
    }

    public double momentum() {
        return momentum;
    }

    public double weightDecay() {
        return weightDecay;
    }

    public boolean nesterov() {
        return nesterov;
    }

    /** Builder mirroring the PyTorch keyword arguments used by the vertical slice. */
    public static final class Builder {
        private List<Parameter> parameters = List.of();
        private double learningRate = 1e-3;
        private double momentum;
        private double weightDecay;
        private double dampening;
        private boolean nesterov;

        private Builder() {}

        public Builder parameters(List<Parameter> value) {
            this.parameters = new ArrayList<>(value);
            return this;
        }

        public Builder learningRate(double value) {
            this.learningRate = value;
            return this;
        }

        public Builder momentum(double value) {
            this.momentum = value;
            return this;
        }

        public Builder weightDecay(double value) {
            this.weightDecay = value;
            return this;
        }

        public Builder dampening(double value) {
            this.dampening = value;
            return this;
        }

        public Builder nesterov(boolean value) {
            this.nesterov = value;
            return this;
        }

        public SGD build() {
            return new SGD(this);
        }
    }
}
