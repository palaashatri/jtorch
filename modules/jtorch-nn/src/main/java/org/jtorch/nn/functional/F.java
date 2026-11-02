package org.jtorch.nn.functional;

import org.jtorch.core.Tensor;

/** Stateless functional forms of the common neural-network operations. */
public final class F {
    private F() {}

    public enum Reduction {
        MEAN("mean"),
        SUM("sum"),
        NONE("none");

        private final String name;

        Reduction(String name) {
            this.name = name;
        }

        public String label() {
            return name;
        }
    }

    public static Tensor linear(Tensor input, Tensor weight) {
        return linear(input, weight, null);
    }

    public static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        return input.linear(weight, bias);
    }

    public static Tensor relu(Tensor input) {
        return input.relu();
    }

    public static Tensor sigmoid(Tensor input) {
        return input.sigmoid();
    }

    public static Tensor tanh(Tensor input) {
        return input.tanh();
    }

    public static Tensor softmax(Tensor input, int dim) {
        return input.softmax(dim);
    }

    public static Tensor logSoftmax(Tensor input, int dim) {
        return input.logSoftmax(dim);
    }

    public static Tensor mseLoss(Tensor input, Tensor target, Reduction reduction) {
        return input.mseLoss(target, reduction.label());
    }

    public static Tensor crossEntropy(Tensor input, Tensor target, Reduction reduction) {
        return input.crossEntropy(target, reduction.label());
    }
}
