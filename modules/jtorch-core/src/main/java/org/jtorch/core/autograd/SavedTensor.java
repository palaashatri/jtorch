package org.jtorch.core.autograd;

import org.jtorch.core.AutogradException;
import org.jtorch.core.Tensor;

/** A tensor saved for backward together with the storage version it must still have. */
public final class SavedTensor {
    private final Tensor tensor;
    private final long version;
    private final String owner;

    SavedTensor(Tensor tensor, String owner) {
        this.tensor = tensor;
        this.version = tensor.impl().version().current();
        this.owner = owner;
    }

    public Tensor unpack() {
        validate();
        return tensor;
    }

    public void validate() {
        if (tensor.impl().version().current() != version) {
            throw new AutogradException("one of the variables needed for gradient computation has been modified by an "
                    + "inplace operation: saved tensor for '" + owner + "' has version " + version
                    + " but storage is now at version " + tensor.impl().version().current());
        }
    }
}
