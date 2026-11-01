package org.jtorch.core.autograd;

import java.util.ArrayList;
import java.util.List;
import org.jtorch.core.Tensor;

/**
 * A node in the reverse-mode autograd graph. Backward formulas are registered per operator by the autograd module
 * and create one node per differentiable forward call.
 */
public abstract class Node {
    private static final Tensor[] NO_TENSORS = new Tensor[0];

    private final String name;
    private final int outputCount;
    private final List<SavedTensor> saved = new ArrayList<>();
    private Tensor[] inputs = NO_TENSORS;
    private Tensor[] outputs = NO_TENSORS;

    protected Node(String name, int outputCount) {
        this.name = name;
        this.outputCount = outputCount;
    }

    public final String name() {
        return name;
    }

    public final int outputCount() {
        return outputCount;
    }

    /** Saves a tensor for backward and records its version counter. */
    protected final <T extends Tensor> T save(T tensor) {
        saved.add(new SavedTensor(tensor, name));
        return tensor;
    }

    /** Retrieves a saved tensor, failing if in-place mutation invalidated it. */
    protected final Tensor saved(int index) {
        return saved.get(index).unpack();
    }

    public final void validateSavedTensors() {
        for (SavedTensor tensor : saved) {
            tensor.validate();
        }
    }

    /** Records the forward inputs so the engine can route gradients and leaf accumulation. */
    public final void setInputs(Tensor... tensors) {
        this.inputs = tensors.clone();
    }

    public final Tensor[] inputs() {
        return inputs;
    }

    /** Records the forward outputs so the engine can allocate missing output gradients and free the graph. */
    public final void setOutputs(Tensor... tensors) {
        if (tensors.length != outputCount) {
            throw new IllegalArgumentException(
                    name + " declared " + outputCount + " outputs but produced " + tensors.length);
        }
        this.outputs = tensors.clone();
    }

    public final Tensor[] outputs() {
        return outputs;
    }

    /**
     * Computes gradients for {@link #inputs()}. {@code gradients[i]} is the accumulated gradient of output
     * {@code i}; entries may be null and the returned array may contain nulls for inputs that do not receive a
     * gradient.
     */
    public abstract Tensor[] backward(Tensor[] gradients);
}
