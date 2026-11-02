package org.jtorch.vision.models;

import java.util.List;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.nn.Conv2d;
import org.jtorch.nn.LayerNorm;
import org.jtorch.nn.Linear;
import org.jtorch.nn.Module;
import org.jtorch.nn.Parameter;
import org.jtorch.nn.Sequential;
import org.jtorch.nn.TransformerEncoderLayer;
import org.jtorch.api.Torch;

/**
 * A compact Vision Transformer fixture: patch embedding, class token, learned positional embedding, a stack of
 * post-norm encoder layers, and a linear classification head. Names mirror PyTorch's ViT parameter tree.
 */
public final class ViT extends Module {
    private final Conv2d patchEmbed;
    private final Parameter classToken;
    private final Parameter posEmbed;
    private final Sequential blocks;
    private final LayerNorm norm;
    private final Linear head;
    private final int imageSize;
    private final int patchSize;

    public ViT(int imageSize, int patchSize, int dim, int heads, int depth, int mlpDim, int classes) {
        if (imageSize <= 0 || patchSize <= 0 || imageSize % patchSize != 0) {
            throw new IllegalArgumentException("ViT imageSize must be divisible by patchSize");
        }
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.patchEmbed = registerModule("patchEmbed", new Conv2d(3, dim,
                new int[] {patchSize, patchSize}, new int[] {patchSize, patchSize}, new int[] {0, 0},
                new int[] {1, 1}, 1, true));
        this.classToken = registerParameter("classToken",
                new Parameter(Torch.randn(Shape.of(1, 1, dim)).mul(0.02)));
        int tokens = (imageSize / patchSize) * (imageSize / patchSize);
        this.posEmbed = registerParameter("posEmbed",
                new Parameter(Torch.randn(Shape.of(1, tokens + 1, dim)).mul(0.02)));
        TransformerEncoderLayer[] layers = new TransformerEncoderLayer[depth];
        for (int i = 0; i < depth; i++) {
            layers[i] = new TransformerEncoderLayer(dim, heads, mlpDim, "gelu");
        }
        this.blocks = registerModule("blocks", new Sequential(layers));
        this.norm = registerModule("norm", new LayerNorm(dim));
        this.head = registerModule("head", new Linear(dim, classes));
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.dim() != 4 || input.size(1) != 3 || input.size(2) != imageSize
                || input.size(3) != imageSize) {
            throw new org.jtorch.core.ShapeException("ViT expects [batch, 3, " + imageSize + ", " + imageSize
                    + "] input, got " + input.shape());
        }
        Tensor patches = patchEmbed.call(input);
        Tensor tokens = patches.flatten(2, 3).transpose(1, 2);
        Tensor cls = classToken.expand(input.size(0), 1, tokens.size(2));
        Tensor sequence = Tensor.cat(List.of(cls, tokens), 1).add(posEmbed);
        sequence = blocks.call(sequence);
        sequence = norm.call(sequence);
        return head.call(sequence.select(1, 0));
    }

    public int imageSize() {
        return imageSize;
    }

    public int patchSize() {
        return patchSize;
    }
}
