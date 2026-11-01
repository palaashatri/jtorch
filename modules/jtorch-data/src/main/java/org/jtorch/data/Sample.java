package org.jtorch.data;

import org.jtorch.core.Tensor;

/** One dataset row: an ordered tuple of tensor columns. */
public record Sample(Tensor[] tensors) {
    public Sample {
        tensors = tensors.clone();
    }

    @Override
    public Tensor[] tensors() {
        return tensors.clone();
    }

    public Tensor tensor(int index) {
        return tensors[index];
    }

    public int arity() {
        return tensors.length;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Sample sample && java.util.Arrays.equals(tensors, sample.tensors);
    }

    @Override
    public int hashCode() {
        return java.util.Arrays.hashCode(tensors);
    }

    @Override
    public String toString() {
        return "Sample" + java.util.Arrays.toString(tensors);
    }
}
