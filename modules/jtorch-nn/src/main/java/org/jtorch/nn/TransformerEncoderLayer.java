package org.jtorch.nn;

import java.util.Locale;
import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;

/** Post-norm transformer encoder layer with self-attention and a feed-forward block, batch-first. */
public final class TransformerEncoderLayer extends Module {
    private final MultiheadAttention selfAttention;
    private final Linear linear1;
    private final Linear linear2;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final String activation;

    public TransformerEncoderLayer(int dModel, int nhead, int dimFeedforward) {
        this(dModel, nhead, dimFeedforward, "relu");
    }

    public TransformerEncoderLayer(int dModel, int nhead, int dimFeedforward, String activation) {
        String normalized = activation.toLowerCase(Locale.ROOT);
        if (!normalized.equals("relu") && !normalized.equals("gelu")) {
            throw new IllegalArgumentException("TransformerEncoderLayer activation must be relu or gelu");
        }
        this.selfAttention = registerModule("self_attn", new MultiheadAttention(dModel, nhead));
        this.linear1 = registerModule("linear1", new Linear(dModel, dimFeedforward));
        this.linear2 = registerModule("linear2", new Linear(dimFeedforward, dModel));
        this.norm1 = registerModule("norm1", new LayerNorm(dModel));
        this.norm2 = registerModule("norm2", new LayerNorm(dModel));
        this.activation = normalized;
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.dim() != 3 || input.size(2) != selfAttention.embedDim()) {
            throw new org.jtorch.core.ShapeException(
                    "TransformerEncoderLayer expects [batch, sequence, dModel] input");
        }
        Tensor attended = input.add(selfAttention.call(input));
        Tensor normalized = norm1.call(attended);
        Tensor hidden = linear1.call(normalized);
        hidden = activation.equals("gelu") ? hidden.gelu() : hidden.relu();
        Tensor fed = normalized.add(linear2.call(hidden));
        return norm2.call(fed);
    }
}
