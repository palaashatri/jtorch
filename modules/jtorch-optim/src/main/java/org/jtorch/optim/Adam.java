package org.jtorch.optim;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import org.jtorch.api.Torch;
import org.jtorch.core.Tensor;
import org.jtorch.nn.Parameter;

/**
 * Adam with PyTorch-compatible moments and bias correction. {@link AdamW} enables decoupled weight decay by
 * construction; this class applies L2 weight decay to the gradient like {@code torch.optim.Adam}.
 */
public class Adam extends Optimizer {
    protected final double beta1;
    protected final double beta2;
    protected final double epsilon;
    protected final double weightDecay;
    protected final boolean decoupled;
    private final Map<Parameter, Tensor> expAvg = new IdentityHashMap<>();
    private final Map<Parameter, Tensor> expAvgSq = new IdentityHashMap<>();
    private final Map<Parameter, Long> steps = new IdentityHashMap<>();

    protected Adam(Config config) {
        super(config.parameters(), config.learningRate());
        if (config.beta1() < 0 || config.beta1() >= 1 || config.beta2() < 0 || config.beta2() >= 1) {
            throw new IllegalArgumentException("Adam betas must be in [0, 1)");
        }
        if (config.epsilon() < 0) {
            throw new IllegalArgumentException("Adam epsilon must be non-negative");
        }
        this.beta1 = config.beta1();
        this.beta2 = config.beta2();
        this.epsilon = config.epsilon();
        this.weightDecay = config.weightDecay();
        this.decoupled = config.decoupled();
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
                if (decoupled && weightDecay != 0) {
                    parameter.mul_(OptimUtil.scalarLike(parameter, 1 - learningRate() * weightDecay));
                } else if (weightDecay != 0) {
                    gradient = gradient.add(parameter, weightDecay);
                }
                long step = steps.merge(parameter, 1L, Long::sum);
                Tensor average = expAvg.computeIfAbsent(parameter, Tensor::zerosLike);
                average.mul_(OptimUtil.scalarLike(average, beta1)).add_(gradient, 1 - beta1);
                Tensor squareAverage = expAvgSq.computeIfAbsent(parameter, Tensor::zerosLike);
                squareAverage.mul_(OptimUtil.scalarLike(squareAverage, beta2)).add_(gradient.mul(gradient), 1 - beta2);

                Tensor correctedAverage = average.div(
                        OptimUtil.scalarLike(parameter, 1 - Math.pow(beta1, step)));
                Tensor correctedSquareAverage = squareAverage.div(
                        OptimUtil.scalarLike(parameter, 1 - Math.pow(beta2, step)));
                Tensor denominator = correctedSquareAverage.sqrt().add(OptimUtil.scalarLike(parameter, epsilon));
                parameter.sub_(correctedAverage.div(denominator), learningRate());
            }
        });
    }

    public double beta1() {
        return beta1;
    }

    public double beta2() {
        return beta2;
    }

    public double epsilon() {
        return epsilon;
    }

    public double weightDecay() {
        return weightDecay;
    }

    /** Immutable optimizer configuration shared by Adam and AdamW. */
    protected record Config(
            List<Parameter> parameters,
            double learningRate,
            double beta1,
            double beta2,
            double epsilon,
            double weightDecay,
            boolean decoupled) {}

    /** Builder for {@code torch.optim.Adam}. */
    public static class Builder {
        private List<Parameter> parameters = List.of();
        private double learningRate = 1e-3;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double epsilon = 1e-8;
        private double weightDecay;

        protected Builder() {}

        public Builder parameters(List<Parameter> value) {
            this.parameters = new ArrayList<>(value);
            return this;
        }

        public Builder learningRate(double value) {
            this.learningRate = value;
            return this;
        }

        public Builder betas(double first, double second) {
            this.beta1 = first;
            this.beta2 = second;
            return this;
        }

        public Builder epsilon(double value) {
            this.epsilon = value;
            return this;
        }

        public Builder weightDecay(double value) {
            this.weightDecay = value;
            return this;
        }

        protected Adam create(Config config) {
            return new Adam(config);
        }

        public Adam build() {
            return create(new Config(parameters, learningRate, beta1, beta2, epsilon, weightDecay, false));
        }
    }
}
