package org.jtorch.nn;

import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;

/** Lookup table from integer indices to dense embedding rows. */
public final class Embedding extends Module {
    private final Parameter weight;
    private final Integer paddingIdx;

    public Embedding(int numEmbeddings, int embeddingDim) {
        this(numEmbeddings, embeddingDim, null);
    }

    public Embedding(int numEmbeddings, int embeddingDim, Integer paddingIdx) {
        if (numEmbeddings <= 0 || embeddingDim <= 0) {
            throw new IllegalArgumentException("Embedding dimensions must be positive");
        }
        if (paddingIdx != null && (paddingIdx < -numEmbeddings || paddingIdx >= numEmbeddings)) {
            throw new IllegalArgumentException("Embedding padding_idx is out of range");
        }
        int normalizedPadding = paddingIdx == null ? -1 : (paddingIdx < 0 ? paddingIdx + numEmbeddings : paddingIdx);
        Tensor initial = Torch.randn(Shape.of(numEmbeddings, embeddingDim));
        if (paddingIdx != null) {
            Torch.noGrad(() -> initial.select(0, normalizedPadding).fill_(0));
        }
        this.weight = registerParameter("weight", new Parameter(initial));
        this.paddingIdx = paddingIdx == null ? null : normalizedPadding;
    }

    @Override
    public Tensor forward(Tensor indices) {
        return weight.embedding(indices, paddingIdx, false, false);
    }

    public Parameter weight() {
        return weight;
    }
}
