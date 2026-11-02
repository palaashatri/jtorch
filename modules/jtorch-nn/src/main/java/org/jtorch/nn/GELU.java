package org.jtorch.nn;

import java.util.Locale;
import org.jtorch.core.Tensor;

/** Gaussian error linear unit with the exact erf form or PyTorch's tanh approximation. */
public final class GELU extends Module {
    private final String approximate;

    public GELU() {
        this("none");
    }

    public GELU(String approximate) {
        String normalized = approximate.toLowerCase(Locale.ROOT);
        if (!normalized.equals("none") && !normalized.equals("tanh")) {
            throw new IllegalArgumentException("GELU approximate must be 'none' or 'tanh'");
        }
        this.approximate = normalized;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.gelu(approximate);
    }
}
